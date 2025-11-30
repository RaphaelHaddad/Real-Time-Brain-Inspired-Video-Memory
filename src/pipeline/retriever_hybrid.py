"""
Hybrid Retrieval: Vector search on chunks + Entity search with graph traversal + Contextual compression
Implements: 1) Parallel vector/fulltext search, 2) Graph hop expansion, 3) Post-compression
"""

import asyncio
import time
import json
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.config import RetrievalConfig
from ..components.neo4j_handler import Neo4jHandler
from ..core.logger import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

logger = get_logger(__name__)


class RerankerError(Exception):
    """Exception type raised when reranker failure should be considered fatal."""
    pass

class HybridRetriever:
    """Hybrid retrieval: chunks (vector) + entities (fulltext) + graph traversal + compression"""

    def __init__(self, config: RetrievalConfig, neo4j_handler: Neo4jHandler, schedule_path: Optional[str], realtime_output: bool = True):
        self.config = config
        self.neo4j_handler = neo4j_handler
        self.schedule_path = schedule_path
        self.retrieval_schedule = self._load_retrieval_schedule() if schedule_path else []
        self.executed_queries = []
        self.executed_query_keys = set()  # Track (query, scheduled_time) tuples to prevent duplicates
        self.output_file = Path("retrieval_results.json") if realtime_output else None
        if self.output_file and realtime_output:
            self.output_file.write_text("[]\n")
        
        # Initialize embeddings for post-compression
        if config.post_compression:
            self.embedder = OpenAIEmbeddings(
                model=neo4j_handler.kg_config.embedding_model,
                openai_api_key=neo4j_handler.kg_config.embedding_api_key,
                openai_api_base=neo4j_handler.kg_config.embedding_endpoint
            )
        else:
            self.embedder = None
        
        logger.info(f"Initialized HybridRetriever: chunks_top_k={config.top_k_chunks}, "
                   f"entities_top_k={config.top_k_entities}, graph_hops={config.graph_hops}, "
                   f"post_compression={config.post_compression}")

    def _load_retrieval_schedule(self) -> List[Dict[str, str]]:
        """Load retrieval schedule from JSON file"""
        try:
            with open(self.schedule_path, 'r') as f:
                schedule_data = json.load(f)
                return schedule_data.get('queries', schedule_data) if isinstance(schedule_data, dict) else schedule_data
        except Exception as e:
            logger.error(f"Error loading retrieval schedule: {str(e)}")
            return []

    def _write_result_realtime(self, result: Dict):
        """Write retrieval result to file in real-time"""
        try:
            if not self.output_file:
                return
            existing = json.loads(self.output_file.read_text()) if self.output_file.exists() else []
            existing.append(result)
            self.output_file.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.error(f"Failed to write real-time result: {e}")

    async def check_and_run_queries(self, current_video_time: str) -> List[Dict]:
        """Check and execute scheduled queries at current time (skip duplicates)"""
        logger.info(f"Checking for queries at batch end time: {current_video_time}")
        results = []

        for query_schedule in self.retrieval_schedule:
            scheduled_time = query_schedule.get('time', '')
            
            # Handle time ranges
            if '-' in current_video_time:
                end_time = current_video_time.split('-')[1].strip()
                start_time = current_video_time.split('-')[0].strip()
                time_matches = scheduled_time <= end_time
            else:
                end_time = current_video_time
                time_matches = scheduled_time == end_time
            
            if time_matches:
                query = query_schedule.get('query', '')
                groundtruth = query_schedule.get('groundtruth', '')
                # Optional true_chunks provided in schedule: allow list or comma-separated string
                true_chunks_schedule = query_schedule.get('true_chunks') or query_schedule.get('true_chunk')
                parsed_true_chunks = None
                if true_chunks_schedule:
                    try:
                        if isinstance(true_chunks_schedule, list):
                            parsed_true_chunks = [int(x) for x in true_chunks_schedule]
                        elif isinstance(true_chunks_schedule, str):
                            # Allow '2,6,40' style
                            parts = [p.strip() for p in true_chunks_schedule.strip('[]').split(',') if p.strip()]
                            parsed_true_chunks = [int(x) for x in parts]
                    except Exception as e:
                        logger.debug(f"Could not parse true_chunks from schedule: {e}")
                
                # Skip if this (query, scheduled_time) was already executed
                query_key = (query, scheduled_time)
                if query_key in self.executed_query_keys:
                    logger.debug(f"Skipping duplicate query: '{query}' at {scheduled_time}")
                    continue
                
                logger.info(f"🔍 RETRIEVAL TRIGGERED at {end_time}")
                logger.info(f"   Query: {query}")
                
                query_start = time.perf_counter()
                try:
                    retrieval_result, _ = await self._perform_hybrid_retrieval(query, parsed_true_chunks)
                    query_time = time.perf_counter() - query_start
                    
                    result = {
                        "time": current_video_time,
                        "query": query,
                        "groundtruth": groundtruth,
                        "retrieval": retrieval_result,
                        "retrieval_time": query_time
                    }
                    results.append(result)
                    self.executed_queries.append(result)
                    self.executed_query_keys.add(query_key)  # Mark as executed
                    self._write_result_realtime(result)
                    
                    logger.info(f"✓ Retrieval completed in {query_time:.3f}s")
                    logger.info(f"   Result: {str(retrieval_result)[:100]}...")
                
                except Exception as e:
                    logger.error(f"Error in retrieval: {str(e)}")

        return results

    async def _perform_hybrid_retrieval(self, query: str, true_chunks: Optional[List[int]] = None) -> tuple[str, bool]:
        """
        Perform hybrid retrieval:
        1. Vector search on chunks (top_k_chunks) or skip if entity_first
        2. Fulltext search on entities (top_k_entities)
        3. Graph traversal from entities (graph_hops)
        4. Post-compression (if enabled and not entity_first)
        5. Reranking (after traversal if rerank_after_traversal, else after vector)
        
        Returns:
            tuple: (result_text, reranking_performed) where reranking_performed is True if reranking was attempted and succeeded
        """
        try:
            logger.debug(f"Starting hybrid retrieval for query: '{query}'")
            retrieval_start = time.perf_counter()
            reranking_performed = False
            
            async with self.neo4j_handler.driver.session() as session:
                # Step 1: Parallel vector + fulltext search (skip vector if entity_first)
                chunk_search_start = time.perf_counter()
                if self.config.entity_first:
                    chunk_results = []
                    entity_results = await self._fulltext_search_entities(session, query)
                else:
                    chunk_results, entity_results = await asyncio.gather(
                        self._vector_search_chunks(session, query),
                        self._fulltext_search_entities(session, query)
                    )
                search_time = time.perf_counter() - chunk_search_start
                logger.debug(f"Search complete: {len(chunk_results)} chunks, {len(entity_results)} entities in {search_time:.3f}s")

                # If true_chunks provided: compute initial vector rankings for them
                if true_chunks:
                    try:
                        # Build mapping from provided index -> ranking position in chunk_results
                        initial_rankings = {}
                        for idx in true_chunks:
                            initial_rankings[idx] = None

                        for pos, c in enumerate(chunk_results, start=1):
                            # Attempt to parse the index from the chunk id (last suffix)
                            try:
                                last_part = str(c.get('id')).split('_')[-1]
                                parsed_idx = int(last_part)
                            except Exception:
                                parsed_idx = None

                            if parsed_idx is not None and parsed_idx in initial_rankings and initial_rankings[parsed_idx] is None:
                                initial_rankings[parsed_idx] = pos

                        # Log initial rankings
                        for idx, pos in initial_rankings.items():
                            if pos is not None:
                                logger.info(f"True chunk {idx} found in initial vector search at rank: {pos}")
                            else:
                                logger.info(f"True chunk {idx} NOT found in initial vector search top_k={self.config.top_k_chunks}")
                    except Exception as e:
                        logger.debug(f"Failed to compute initial true_chunks rankings: {e}")
                
                # Step 2: Graph traversal from top entities
                graph_start = time.perf_counter()
                expanded_entities, traversal_chunks, traversal_relationships = await self._expand_entity_graph_with_chunks(session, entity_results)
                graph_time = time.perf_counter() - graph_start
                logger.debug(f"Graph expansion: {len(expanded_entities)} entities, {len(traversal_chunks)} chunks, {len(traversal_relationships)} relationships after {self.config.graph_hops} hops in {graph_time:.3f}s")
                
                # Step 3: Post-compression (if enabled and not entity_first)
                if self.config.post_compression and chunk_results and not self.config.entity_first:
                    compress_start = time.perf_counter()
                    chunk_results = await self._post_compress_chunks(query, chunk_results)
                    compress_time = time.perf_counter() - compress_start
                    logger.debug(f"Post-compression: {len(chunk_results)} chunks retained in {compress_time:.3f}s")
                
                # Step 4: Reranking
                if self.config.rerank_after_traversal:
                    rerank_start = time.perf_counter()
                    # Rerank traversal results
                    if self.config.rerank_entities and expanded_entities:
                        expanded_entities = await self._rerank_entities(query, expanded_entities, raise_on_failure=True)
                        reranking_performed = True
                    if self.config.rerank_relationships and traversal_relationships:
                        traversal_relationships = await self._rerank_relationships(query, traversal_relationships, raise_on_failure=True)
                        reranking_performed = True
                    if traversal_chunks:
                        traversal_chunks = await self._rerank_chunks(query, traversal_chunks, raise_on_failure=True)
                        reranking_performed = True
                    rerank_time = time.perf_counter() - rerank_start
                    logger.debug(f"Post-traversal reranking: entities={len(expanded_entities)}, relationships={len(traversal_relationships)}, chunks={len(traversal_chunks)} in {rerank_time:.3f}s")
                elif self.config.use_reranker and chunk_results:
                    rerank_start = time.perf_counter()
                    chunk_results = await self._rerank_chunks(query, chunk_results)
                    reranking_performed = True
                    rerank_time = time.perf_counter() - rerank_start
                    logger.debug(f"Vector reranking: complete in {rerank_time:.3f}s")
                
                # Format final results
                result_chunks = (chunk_results or []) + (traversal_chunks or [])
                
                # Limit final chunks to top_k_chunks
                result_chunks = result_chunks[:self.config.top_k_chunks]

                # If true_chunks provided: compute final rankings and missing ones
                if true_chunks:
                    try:
                        final_rankings = {idx: None for idx in true_chunks}
                        seen_chunk_ids = set()
                        for pos, c in enumerate(result_chunks, start=1):
                            cid = str(c.get('id'))
                            seen_chunk_ids.add(cid)
                            try:
                                last_part = cid.split('_')[-1]
                                parsed_idx = int(last_part)
                            except Exception:
                                parsed_idx = None

                            if parsed_idx is not None and parsed_idx in final_rankings and final_rankings[parsed_idx] is None:
                                final_rankings[parsed_idx] = pos

                        # Log final rankings and missing
                        for idx, pos in final_rankings.items():
                            if pos is not None:
                                logger.info(f"True chunk {idx} found among final retrieval candidates at rank: {pos}")
                            else:
                                # Not in final candidates; check if it was not even chosen in graph exploration
                                logger.info(f"True chunk {idx} NOT found among final retrieval candidates (not included in chunk_results+traversal_chunks)")
                                # Optional: check if chunk id exists at all in seen ids via suffix check
                                logger.debug(f"True chunk {idx} was not present in final candidates' ids: {list(seen_chunk_ids)[:10]}...")
                    except Exception as e:
                        logger.debug(f"Failed to compute final true_chunks rankings: {e}")

                result_text = self._format_retrieval_results(query, result_chunks, expanded_entities, traversal_relationships)
                
                total_time = time.perf_counter() - retrieval_start
                logger.debug(f"Total retrieval time: {total_time:.3f}s")
                return result_text, reranking_performed

        except RerankerError:
            # Reranker failure when strict mode is enabled should be raised to the caller
            raise
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {str(e)}")
            return f"Retrieval failed: {str(e)}", False

    async def _vector_search_chunks(self, session, query: str) -> List[Dict[str, Any]]:
        """Vector similarity search on chunk embeddings"""
        try:
            logger.debug(f"Vector search on chunks with top_k={self.config.top_k_chunks}")
            
            # Get query embedding
            query_embedding = await self.neo4j_handler.embedder.aembed_query(query)
            
            # Cypher: Vector similarity search on chunks
            result = await session.run(
                """
                MATCH (c:Chunk {graph_uuid: $graph_uuid})
                WHERE c.embedding IS NOT NULL
                WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS similarity
                WHERE similarity > 0.3
                RETURN c.id AS chunk_id, c.time AS chunk_time, c.content AS content, similarity AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                graph_uuid=self.neo4j_handler.run_uuid,
                query_embedding=query_embedding,
                limit=self.config.top_k_chunks
            )
            
            chunks = []
            async for record in result:
                chunks.append({
                    "id": record["chunk_id"],
                    "time": record["chunk_time"],
                    "content": record["content"],
                    "score": float(record["score"]),
                    "source": "vector"
                })
            
            logger.debug(f"Vector search returned {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _fulltext_search_entities(self, session, query: str) -> List[Dict[str, Any]]:
        """Fulltext search on entity names"""
        try:
            logger.debug(f"Fulltext search on entities with top_k={self.config.top_k_entities}")
            
            result = await session.run(
                """
                CALL db.index.fulltext.queryNodes("entityName", $search_query)
                YIELD node, score
                WHERE node.graph_uuid = $graph_uuid
                RETURN node.name AS name, node.batch_time AS batch_time, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                search_query=query,
                graph_uuid=self.neo4j_handler.run_uuid,
                limit=self.config.top_k_entities
            )
            
            entities = []
            async for record in result:
                entities.append({
                    "name": record["name"],
                    "batch_time": record["batch_time"] or "",
                    "score": float(record["score"]),
                    "source": "fulltext"
                })
            
            logger.debug(f"Fulltext search returned {len(entities)} entities")
            return entities
        
        except Exception as e:
            logger.warning(f"Fulltext search failed: {e}")
            return []

    async def _expand_entity_graph(self, session, entities: List[Dict], hops: int = None) -> List[Dict]:
        """Expand entity graph via relationship traversal"""
        if not entities or not hops:
            hops = self.config.graph_hops
        
        try:
            logger.debug(f"Expanding entity graph with {hops} hops")
            expanded = set()
            
            for entity in entities:
                # Traverse relationships from this entity
                result = await session.run(
                    f"""
                    MATCH (e:Entity {{name: $entity_name, graph_uuid: $graph_uuid}})
                    MATCH (e)-[*1..{hops}]-(related:Entity)
                    WHERE related.graph_uuid = $graph_uuid
                    RETURN DISTINCT related.name AS name, related.batch_time AS batch_time
                    LIMIT 20
                    """,
                    entity_name=entity["name"],
                    graph_uuid=self.neo4j_handler.run_uuid
                )
                
                async for record in result:
                    expanded.add((record["name"], record["batch_time"] or ""))
            
            # Convert back to list
            expanded_list = [{"name": n, "batch_time": t, "source": "graph_traversal"} for n, t in expanded]
            logger.debug(f"Graph expansion found {len(expanded_list)} related entities")
            return expanded_list
        
        except Exception as e:
            logger.warning(f"Graph expansion failed: {e}")
            return []

    async def _expand_entity_graph_with_chunks(self, session, entities: List[Dict], hops: int = None) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Expand entity graph via relationship traversal, collecting entities, chunks, and relationships"""
        if not entities or not hops:
            hops = self.config.graph_hops
        
        try:
            logger.debug(f"Expanding entity graph with {hops} hops")
            expanded_entities = set()
            traversal_chunks = set()
            traversal_relationships = set()
            
            for entity in entities:
                # Traverse relationships from this entity, collecting entities, chunks, and relationships
                result = await session.run(
                    f"""
                    MATCH (e:Entity {{name: $entity_name, graph_uuid: $graph_uuid}})
                    MATCH path = (e)-[*1..{hops}]-(related)
                    WHERE related.graph_uuid = $graph_uuid
                    UNWIND relationships(path) AS rel
                    RETURN DISTINCT 
                        related.name AS related_name, 
                        related.id AS related_id,
                        related.batch_time AS related_batch_time,
                        labels(related) AS related_labels,
                        type(rel) AS rel_type,
                        startNode(rel).name AS start_name,
                        endNode(rel).name AS end_name,
                        properties(rel) AS rel_props
                    """,
                    entity_name=entity["name"],
                    graph_uuid=self.neo4j_handler.run_uuid
                )
                
                async for record in result:
                    related_labels = record["related_labels"]
                    if "Entity" in related_labels:
                        expanded_entities.add((record["related_name"], record["related_batch_time"] or ""))
                    elif "Chunk" in related_labels:
                        # Get chunk details
                        chunk_result = await session.run(
                            """
                            MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                            RETURN c.content AS content, c.time AS time
                            """,
                            chunk_id=record["related_id"],
                            graph_uuid=self.neo4j_handler.run_uuid
                        )
                        async for chunk_record in chunk_result:
                            traversal_chunks.add((
                                record["related_id"],
                                chunk_record["content"],
                                chunk_record["time"]
                            ))
                    
                    # Collect relationships
                    rel_desc = f"{record['start_name']} -[{record['rel_type']}]-> {record['end_name']}"
                    traversal_relationships.add(rel_desc)
            
            # Convert to lists
            expanded_list = [{"name": n, "batch_time": t, "source": "graph_traversal"} for n, t in expanded_entities]
            chunks_list = [{"id": cid, "content": content, "time": time, "source": "graph_traversal"} for cid, content, time in traversal_chunks]
            relationships_list = [{"description": desc, "source": "graph_traversal"} for desc in traversal_relationships]
            
            logger.debug(f"Graph expansion found {len(expanded_list)} entities, {len(chunks_list)} chunks, {len(relationships_list)} relationships")
            return expanded_list, chunks_list, relationships_list
        
        except Exception as e:
            logger.warning(f"Graph expansion failed: {e}")
            return [], [], []

    async def _post_compress_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Post-compression: split chunks and filter by similarity"""
        if not self.embedder or not chunks:
            return chunks
        
        try:
            logger.debug(f"Post-compression: splitting {len(chunks)} chunks")
            
            # Get query embedding
            query_embedding = await self.embedder.aembed_query(query)
            query_embedding_vector = query_embedding  # Use raw embedding
            
            # Split chunks into smaller segments
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=256,
                chunk_overlap=32,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            compressed_chunks = []
            for chunk in chunks:
                content = chunk["content"]
                
                # Split chunk into segments
                segments = splitter.split_text(content)
                
                # Embed and score each segment
                for segment in segments:
                    try:
                        segment_embedding = await self.embedder.aembed_query(segment)
                        
                        # Compute cosine similarity
                        similarity = self._cosine_similarity(query_embedding_vector, segment_embedding)
                        
                        if similarity >= self.config.compression_threshold:
                            compressed_chunks.append({
                                **chunk,
                                "content": segment,  # Keep only high-similarity segments
                                "compression_score": float(similarity)
                            })
                    except Exception as e:
                        logger.debug(f"Failed to embed segment: {e}")
                        continue
            
            logger.debug(f"Post-compression retained {len(compressed_chunks)}/{len(chunks)} segments")
            return compressed_chunks[:self.config.top_k]  # Limit final output
        
        except Exception as e:
            logger.warning(f"Post-compression failed: {e}")
            return chunks

    async def _rerank_chunks(self, query: str, chunks: List[Dict], raise_on_failure: bool = False) -> List[Dict]:
        """Rerank chunks using external reranker"""
        if not self.config.use_reranker or not chunks:
            return chunks

        try:
            logger.debug(f"Reranking {len(chunks)} chunks")
            
            documents = [c["content"] for c in chunks]
            payload = {
                "query": query,
                "documents": documents,
                "top_k": len(documents)
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.reranker_api_key:
                headers["Authorization"] = f"Bearer {self.config.reranker_api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.config.reranker_endpoint,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                rerank_result = response.json()
            
            # Reorder chunks based on reranker
            if "results" in rerank_result and isinstance(rerank_result["results"], list):
                reranked_indices = [item["index"] for item in rerank_result["results"]]
                return [chunks[i] for i in reranked_indices if i < len(chunks)]

            # Unexpected response format
            msg = "Reranker returned unexpected format for chunks"
            if raise_on_failure:
                raise RerankerError(msg)
            logger.warning(msg)
            return chunks
        
        except Exception as e:
            if raise_on_failure:
                logger.error(f"Reranking (chunks) failed and raise_on_failure=True: {e}")
                raise RerankerError(str(e))
            logger.warning(f"Reranking failed: {e}")
            return chunks

    async def _rerank_entities(self, query: str, entities: List[Dict], raise_on_failure: bool = False) -> List[Dict]:
        """Rerank entities using external reranker"""
        if not self.config.use_reranker or not entities:
            return entities

        try:
            logger.debug(f"Reranking {len(entities)} entities")
            
            documents = [e["name"] for e in entities]
            payload = {
                "query": query,
                "documents": documents,
                "top_k": len(documents)
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.reranker_api_key:
                headers["Authorization"] = f"Bearer {self.config.reranker_api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.config.reranker_endpoint,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                rerank_result = response.json()
            
            # Reorder entities based on reranker
            if "results" in rerank_result and isinstance(rerank_result["results"], list):
                reranked_indices = [item["index"] for item in rerank_result["results"]]
                return [entities[i] for i in reranked_indices if i < len(entities)]

            msg = "Reranker returned unexpected format for entities"
            if raise_on_failure:
                raise RerankerError(msg)
            logger.warning(msg)
            return entities
        
        except Exception as e:
            if raise_on_failure:
                logger.error(f"Reranking (entities) failed and raise_on_failure=True: {e}")
                raise RerankerError(str(e))
            logger.warning(f"Entity reranking failed: {e}")
            return entities

    async def _rerank_relationships(self, query: str, relationships: List[Dict], raise_on_failure: bool = False) -> List[Dict]:
        """Rerank relationships using external reranker"""
        if not self.config.use_reranker or not relationships:
            return relationships

        try:
            logger.debug(f"Reranking {len(relationships)} relationships")
            
            documents = [r["description"] for r in relationships]
            payload = {
                "query": query,
                "documents": documents,
                "top_k": len(documents)
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.reranker_api_key:
                headers["Authorization"] = f"Bearer {self.config.reranker_api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.config.reranker_endpoint,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                rerank_result = response.json()
            
            # Reorder relationships based on reranker
            if "results" in rerank_result and isinstance(rerank_result["results"], list):
                reranked_indices = [item["index"] for item in rerank_result["results"]]
                return [relationships[i] for i in reranked_indices if i < len(relationships)]

            msg = "Reranker returned unexpected format for relationships"
            if raise_on_failure:
                raise RerankerError(msg)
            logger.warning(msg)
            return relationships
        
        except Exception as e:
            if raise_on_failure:
                logger.error(f"Reranking (relationships) failed and raise_on_failure=True: {e}")
                raise RerankerError(str(e))
            logger.warning(f"Relationship reranking failed: {e}")
            return relationships

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 * mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def _format_retrieval_results(self, query: str, chunks: List[Dict], entities: List[Dict], relationships: List[Dict] = None) -> str:
        """Format retrieval results for output"""
        parts = []
        
        if chunks:
            parts.append(f"Chunks ({len(chunks)}):")
            for i, chunk in enumerate(chunks, 1):
                score_info = f" (score: {chunk.get('score', 0):.3f})" if 'score' in chunk else ""
                time_info = f" [time: {chunk.get('time', '')}]" if chunk.get('time') else ""
                # Include full chunk content for benchmarking (no truncation)
                parts.append(f"  {i}. {chunk['content']}{score_info}{time_info}")
        
        if entities:
            parts.append(f"\nRelated Entities ({len(entities)}):")
            for i, ent in enumerate(entities[:self.config.top_k_entities], 1):
                score_info = f" (score: {ent.get('score', 0):.3f})" if 'score' in ent else ""
                parts.append(f"  {i}. {ent['name']}{score_info}")
        
        if relationships:
            parts.append(f"\nRelated Relationships ({len(relationships)}):")
            for i, rel in enumerate(relationships[:self.config.top_k_relationships], 1):
                parts.append(f"  {i}. {rel['description']}")
        
        if not parts:
            return f"No results found for query '{query}'"
        
        return f"Found results for '{query}':\n" + "\n".join(parts)

    async def close(self):
        """Close retriever"""
        logger.info("Closing hybrid retriever")
