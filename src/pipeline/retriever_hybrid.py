"""
Hybrid Retrieval: Vector search on chunks + Entity search with graph traversal + Contextual compression
Implements: 1) Parallel vector/fulltext search, 2) Graph hop expansion, 3) Post-compression
            4) Community-based retrieval (optional)
"""

import asyncio
import time
import json
import httpx
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.config import RetrievalConfig, CommunityHighGraphConfig
from ..components.neo4j_handler import Neo4jHandler
from ..core.logger import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

logger = get_logger(__name__)


class RerankerError(Exception):
    """Exception type raised when reranker failure should be considered fatal."""
    pass

class HybridRetriever:
    """Hybrid retrieval: chunks (vector) + entities (fulltext) + graph traversal + compression + community retrieval"""

    def __init__(self, config: RetrievalConfig, neo4j_handler: Neo4jHandler, schedule_path: Optional[str], 
                 realtime_output: bool = True, community_config: Optional[CommunityHighGraphConfig] = None):
        self.config = config
        self.neo4j_handler = neo4j_handler
        self.schedule_path = schedule_path
        self.community_config = community_config
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
        
        # Log community retrieval status
        community_retriever_enabled = community_config and community_config.community_retriever
        logger.info(f"Initialized HybridRetriever: chunks_top_k={config.top_k_chunks}, "
                   f"entities_top_k={config.top_k_entities}, graph_hops={config.graph_hops}, "
                   f"post_compression={config.post_compression}, "
                   f"community_retriever={community_retriever_enabled}")

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
                    retrieval_result, _, _ = await self._perform_hybrid_retrieval(query, parsed_true_chunks)
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

    async def _perform_hybrid_retrieval(self, query: str, true_chunks: Optional[List[int]] = None, expected_chunk_ids: Optional[List[int]] = None, analysis_dir: Optional[Path] = None) -> tuple[str, bool, Optional[Path]]:
        """
        Perform hybrid retrieval:
        1. Vector search on chunks (top_k_chunks) or skip if entity_first
           OR community retrieval if community_retriever is enabled
        2. Fulltext search on entities (top_k_entities)
        3. Graph traversal from entities (graph_hops)
        4. Post-compression (if enabled and not entity_first)
        5. Reranking (after traversal if rerank_after_traversal, else after vector)
           NOTE: Community summaries are NOT included in reranking - only chunks/entities
        
        Returns:
            tuple: (result_text, reranking_performed) where reranking_performed is True if reranking was attempted and succeeded
        """
        try:
            logger.debug(f"Starting hybrid retrieval for query: '{query}'")
            retrieval_start = time.perf_counter()
            reranking_performed = False
            analysis_path = None

            # Initialize deep analysis container if requested
            deep_analysis = None
            if analysis_dir and expected_chunk_ids:
                deep_analysis = {
                    "query": query,
                    "expected_chunk_ids": expected_chunk_ids,
                    "stages": {}
                }
            
            # Check if community retrieval is enabled
            use_community_retrieval = (
                self.community_config and 
                self.community_config.community_retriever
            )
            community_traversal_seed = getattr(self.community_config, 'community_traversal_seed', 'chunk_node') if self.community_config else 'entity_node'
            
            # Add traversal seed strategy to deep analysis
            if deep_analysis is not None:
                deep_analysis["traversal_seed_type"] = community_traversal_seed if use_community_retrieval else "entity_node"
            
            async with self.neo4j_handler.driver.session() as session:
                # Step 1: Parallel vector + fulltext search (skip vector if entity_first or using community)
                chunk_search_start = time.perf_counter()
                questions = []
                
                if use_community_retrieval:
                    # Community-based retrieval: search CommunitySummary nodes first
                    logger.info(f"🏘️ Using community-based retrieval (seed from: {community_traversal_seed})")
                    chunk_results, entity_results, community_results, questions, community_info = await self._perform_community_retrieval(
                        query, expected_chunk_ids, collect_questions=bool(deep_analysis)
                    )
                elif self.config.entity_first:
                    chunk_results = []
                    entity_results = await self._fulltext_search_entities(session, query)
                    community_results = []
                else:
                    chunk_results, entity_results = await asyncio.gather(
                        self._vector_search_chunks(session, query),
                        self._fulltext_search_entities(session, query)
                    )
                    community_results = []
                search_time = time.perf_counter() - chunk_search_start
                logger.debug(f"Search complete: {len(chunk_results)} chunks, {len(entity_results)} entities in {search_time:.3f}s")

                if deep_analysis is not None:
                    if use_community_retrieval:
                        if community_info:
                            community_rank_dict, chunk_to_comm = community_info
                            chunk_ranks = []
                            expected_chunk_communities = {}  # Track communities for expected chunks
                            for expected in expected_chunk_ids:
                                comm_id = chunk_to_comm.get(expected)
                                rank = community_rank_dict.get(comm_id) if comm_id else None
                                chunk_ranks.append({
                                    "expected_chunk_id": expected,
                                    "rank": rank,
                                    "chunk_id": None,
                                    "source": "community_rank" if rank else None
                                })
                                if comm_id:
                                    expected_chunk_communities[expected] = comm_id
                        else:
                            chunk_ranks = self._rank_expected_chunks(chunk_results or [], expected_chunk_ids)
                            expected_chunk_communities = {}
                        
                        # Build community descriptions
                        community_descriptions = []
                        for cs in community_results:
                            community_descriptions.append({
                                "community_id": cs.get('community_id'),
                                "score": cs.get('score'),
                                "description": cs.get('content'),
                                "num_chunks": len(cs.get('member_chunk_ids', []))
                            })
                        
                        # Fetch aggregated questions for communities of expected chunks
                        expected_chunk_questions = await self._get_community_questions_for_expected_chunks(expected_chunk_communities)
                        
                        # Collect seed chunk IDs (from community expansion)
                        seed_chunk_ids = [c.get("id") for c in chunk_results] if chunk_results else []
                        
                        deep_analysis["stages"]["community_search"] = {
                            "communities_returned": len(community_results),
                            "retrieved_communities": community_descriptions,
                            "chunk_ranks": chunk_ranks,
                            "seed_chunk_ids": seed_chunk_ids,
                            "seed_type": community_traversal_seed,
                            "expected_chunk_community_questions": expected_chunk_questions,
                            "question_hits": self._collect_question_hits(questions, expected_chunk_ids)
                        }
                    else:
                        deep_analysis["stages"]["vector_search"] = {
                            "chunk_ranks": self._rank_expected_chunks(chunk_results or [], expected_chunk_ids)
                        }

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
                
                # Step 2: Graph traversal (from entities or chunks depending on community_traversal_seed)
                graph_start = time.perf_counter()
                if use_community_retrieval and community_traversal_seed == 'chunk_node':
                    # Seed traversal from community-derived chunks (skip entity collection)
                    logger.debug("🔗 Seeding traversal from community chunk nodes")
                    expanded_entities, traversal_chunks, traversal_relationships = await self._expand_chunk_graph_with_chunks(session, chunk_results)
                else:
                    # Seed traversal from entities (standard path)
                    expanded_entities, traversal_chunks, traversal_relationships = await self._expand_entity_graph_with_chunks(session, entity_results)
                graph_time = time.perf_counter() - graph_start
                logger.debug(f"Graph expansion: {len(expanded_entities)} entities, {len(traversal_chunks)} chunks, {len(traversal_relationships)} relationships after {self.config.graph_hops} hops in {graph_time:.3f}s")

                if deep_analysis is not None:
                    deep_analysis["stages"]["graph_traversal"] = {
                        "chunk_ranks": self._rank_expected_chunks(traversal_chunks or [], expected_chunk_ids)
                    }
                
                # Step 3: Post-compression (if enabled and not entity_first)
                # NOTE: When using chunk-seeded traversal, skip post-compression on seed chunks
                # to preserve community selections; only compress non-seed chunks if needed
                if self.config.post_compression and chunk_results and not self.config.entity_first:
                    if use_community_retrieval and community_traversal_seed == 'chunk_node':
                        # Don't post-compress community seed chunks; they're already selected
                        logger.debug("Skipping post-compression for community seed chunks (chunk-seeded traversal enabled)")
                    else:
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

                if deep_analysis is not None:
                    # Track seed chunks in final results for chunk-seeded traversal
                    seed_chunk_ids_set = set()
                    if use_community_retrieval and community_traversal_seed == 'chunk_node' and chunk_results:
                        seed_chunk_ids_set = {c.get("id") for c in chunk_results if isinstance(c, dict)}
                    
                    final_chunks_with_seed_info = []
                    for chunk in (result_chunks or []):
                        chunk_copy = dict(chunk)
                        chunk_copy["is_seed"] = chunk.get("id") in seed_chunk_ids_set
                        final_chunks_with_seed_info.append(chunk_copy)
                    
                    deep_analysis["stages"]["final"] = {
                        "chunk_ranks": self._rank_expected_chunks(final_chunks_with_seed_info or [], expected_chunk_ids),
                        "seed_chunk_ids_in_final": list(seed_chunk_ids_set)
                    }
                    analysis_path = self._write_deep_analysis(deep_analysis, analysis_dir, query)
                
                total_time = time.perf_counter() - retrieval_start
                logger.debug(f"Total retrieval time: {total_time:.3f}s")
                return result_text, reranking_performed, analysis_path

        except RerankerError:
            # Reranker failure when strict mode is enabled should be raised to the caller
            raise
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {str(e)}")
            return f"Retrieval failed: {str(e)}", False, None

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
                RETURN c.id AS chunk_id, c.time AS chunk_time, c.content AS content, c.original_chunk_id AS original_chunk_id, similarity AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                graph_uuid=self.neo4j_handler.run_uuid,
                query_embedding=query_embedding,
                limit=self.config.top_k_chunks
            )
            
            chunks = []
            async for record in result:
                original_chunk_id = record["original_chunk_id"] if "original_chunk_id" in record.keys() else None
                chunks.append({
                    "id": record["chunk_id"],
                    "time": record["chunk_time"],
                    "content": record["content"],
                    "original_chunk_id": original_chunk_id,
                    "score": float(record["score"]),
                    "source": "vector"
                })
            
            logger.debug(f"Vector search returned {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _get_community_questions_for_expected_chunks(self, expected_chunk_communities: Dict[int, str]) -> Dict[int, List[str]]:
        """
        Fetch aggregated questions for each unique community of expected chunks.
        
        Args:
            expected_chunk_communities: Dict mapping expected_chunk_id -> community_id
            
        Returns:
            Dict mapping expected_chunk_id -> list of aggregated questions from its community
        """
        if not expected_chunk_communities:
            return {}
        
        try:
            result_dict = {}
            unique_communities = set(expected_chunk_communities.values())
            
            async with self.neo4j_handler.driver.session() as session:
                for comm_id in unique_communities:
                    result = await session.run(
                        """
                        MATCH (cs:CommunitySummary {community_id: $comm_id, graph_uuid: $g})
                        RETURN cs.question_chunks AS questions
                        """,
                        comm_id=comm_id,
                        g=self.neo4j_handler.run_uuid
                    )
                    async for record in result:
                        questions = record['questions'] or []
                        # Map this community to all expected chunks that belong to it
                        for exp_chunk_id, c_id in expected_chunk_communities.items():
                            if c_id == comm_id:
                                result_dict[exp_chunk_id] = questions
            
            return result_dict
        except Exception as e:
            logger.warning(f"Error fetching community questions for expected chunks: {e}")
            return {}

    async def _perform_community_retrieval(self, query: str, expected_chunk_ids: Optional[List[int]] = None, collect_questions: bool = False) -> tuple[List[Dict], List[Dict], List[Dict], List[Dict], Optional[tuple[Dict[str, int], Dict[int, str]]]]:
        """
        Perform community-based retrieval:
        1. Vector search on CommunitySummary descriptions
        2. Hop from top communities to their member chunks
        3. Get entities connected to those chunks

        NOTE: Community summaries themselves are returned separately and NOT included in reranking.
        Only chunks and entities from the community hop are reranked.

        Returns:
            tuple: (chunks, entities, community_summaries, questions, community_info)
            where community_info is (community_rank_dict, chunk_to_comm) if expected_chunk_ids else None
        """
        try:
            logger.info(f"🏘️ Community retrieval for query: '{query}'")
            
            # Step 1: Get query embedding
            query_embedding = await self.neo4j_handler.embedder.aembed_query(query)
            
            # Step 2: Search CommunitySummary nodes by vector similarity
            community_summaries = await self.neo4j_handler.vector_search_community_summaries(
                query_embedding=query_embedding,
                top_k=self.config.top_k_chunks,  # Use same top_k as chunks
                min_similarity=0.3
            )
            
            if not community_summaries:
                logger.info("No community summaries found, falling back to standard retrieval")
                return [], [], [], [], None
            
            logger.info(f"Found {len(community_summaries)} relevant communities")
            for i, cs in enumerate(community_summaries[:3], 1):
                logger.debug(f"  Community {i}: {cs['community_id']} (score: {cs['score']:.3f}) - "
                            f"{len(cs['member_chunk_ids'])} chunks")
            
            # If expected_chunk_ids provided, get all community summaries for ranking
            community_info = None
            if expected_chunk_ids:
                all_community_summaries = await self.neo4j_handler.vector_search_community_summaries(
                    query_embedding=query_embedding,
                    top_k=1000,  # large number to get all
                    min_similarity=0.0
                )
                community_rank_dict = {cs['community_id']: i+1 for i, cs in enumerate(all_community_summaries, 1)}
                
                # Get chunk to community mapping for expected chunks
                async with self.neo4j_handler.driver.session() as session:
                    result = await session.run(
                        """
                        MATCH (c:Chunk {graph_uuid: $g})
                        WHERE c.original_chunk_id IN $ocids
                        RETURN c.original_chunk_id AS ocid, c.community_id AS comm_id
                        """,
                        g=self.neo4j_handler.run_uuid,
                        ocids=expected_chunk_ids
                    )
                    chunk_to_comm = {}
                    async for record in result:
                        chunk_to_comm[record['ocid']] = record['comm_id']
                
                community_info = (community_rank_dict, chunk_to_comm)
            
            # Step 3: Collect all member chunk IDs from top communities
            all_chunk_ids = []
            for cs in community_summaries:
                all_chunk_ids.extend(cs.get('member_chunk_ids', []))
            
            # Deduplicate while preserving order (higher-scoring communities first)
            seen_ids = set()
            unique_chunk_ids = []
            for cid in all_chunk_ids:
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    unique_chunk_ids.append(cid)
            
            # Limit to top_k_chunks
            unique_chunk_ids = unique_chunk_ids[:self.config.top_k_chunks * 2]  # Get more for filtering
            
            # Step 4: Get chunk details
            chunks = await self.neo4j_handler.get_chunks_by_ids(unique_chunk_ids)
            logger.debug(f"Retrieved {len(chunks)} chunks from communities")
            
            # Step 5: Get entities connected to these chunks
            entities = await self.neo4j_handler.get_entities_from_chunks(unique_chunk_ids)
            logger.debug(f"Retrieved {len(entities)} entities from community chunks")

            # Optionally collect question nodes for analysis
            questions = []
            if collect_questions and unique_chunk_ids:
                questions = await self.neo4j_handler.get_questions_for_chunk_ids(unique_chunk_ids)
                logger.debug(f"Retrieved {len(questions)} question nodes from community chunks")
            
            return chunks, entities, community_summaries, questions, community_info
            
        except Exception as e:
            logger.error(f"Community retrieval failed: {e}")
            return [], [], [], []

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

    async def _expand_chunk_graph_with_chunks(self, session, chunks: List[Dict], hops: int = None) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Expand chunk graph by seeding from chunk nodes.
        Traverses from chunks → entities → chunks to discover related chunks.
        Uses the configured hop_method (naive BFS, page_rank, or ch3_l3).
        
        Args:
            session: Neo4j session
            chunks: List of chunk dicts with 'id', 'content', etc.
            hops: Number of hops (defaults to config.graph_hops)
        
        Returns:
            tuple: (expanded_entities, traversal_chunks, relationships) from traversal
        """
        hop_method = self.config.hop_method
        logger.info(f"🔍 Using hop method (chunk-seeded): {hop_method}")
        
        if hop_method == "page_rank":
            return await self._expand_chunks_via_pagerank(chunks)
        elif hop_method == "ch3_l3":
            return await self._expand_chunks_via_ch3l3(chunks)
        else:
            # Default: naive BFS traversal from chunks
            return await self._expand_chunk_graph_naive(session, chunks, hops)

    async def _expand_chunk_graph_naive(self, session, chunks: List[Dict], hops: int = None) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Expand chunk graph via naive BFS: start from chunk nodes, traverse to entities, then to related chunks.
        """
        if not chunks or not hops:
            hops = self.config.graph_hops
        
        try:
            logger.debug(f"Expanding chunk graph with {hops} hops (naive BFS from chunks)")
            expanded_entities = set()
            traversal_chunks = set()
            traversal_relationships = set()
            seen_chunk_ids = set()
            
            # Add seed chunks to seen set (avoid re-adding them as traversal results)
            for chunk in chunks:
                if isinstance(chunk, dict):
                    seen_chunk_ids.add(chunk.get("id"))
            
            for chunk in chunks:
                chunk_id = chunk.get("id") if isinstance(chunk, dict) else chunk
                
                # Traverse from this chunk: Chunk -> Entity -> related nodes
                result = await session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                    MATCH (c)-[r]-(related)
                    WHERE related.graph_uuid = $graph_uuid
                    RETURN DISTINCT 
                        related.name AS related_name, 
                        related.id AS related_id,
                        related.batch_time AS related_batch_time,
                        labels(related) AS related_labels,
                        type(r) AS rel_type,
                        startNode(r).name AS start_name,
                        endNode(r).name AS end_name,
                        properties(r) AS rel_props
                    LIMIT 100
                    """,
                    chunk_id=chunk_id,
                    graph_uuid=self.neo4j_handler.run_uuid
                )
                
                async for record in result:
                    related_labels = record["related_labels"]
                    
                    if "Entity" in related_labels:
                        # Found an entity connected to the chunk
                        expanded_entities.add((record["related_name"], record["related_batch_time"] or ""))
                        
                        # Now traverse from this entity to find other chunks
                        entity_name = record["related_name"]
                        entity_result = await session.run(
                            """
                            MATCH (e:Entity {name: $entity_name, graph_uuid: $graph_uuid})
                            MATCH (e)-[r2]-(chunk2:Chunk)
                            WHERE chunk2.graph_uuid = $graph_uuid
                            RETURN DISTINCT 
                                chunk2.id AS chunk_id, 
                                chunk2.content AS content, 
                                chunk2.time AS time,
                                chunk2.original_chunk_id AS original_chunk_id
                            LIMIT 50
                            """,
                            entity_name=entity_name,
                            graph_uuid=self.neo4j_handler.run_uuid
                        )
                        
                        async for entity_record in entity_result:
                            related_chunk_id = entity_record["chunk_id"]
                            # Avoid re-adding seed chunks
                            if related_chunk_id not in seen_chunk_ids:
                                seen_chunk_ids.add(related_chunk_id)
                                traversal_chunks.add((
                                    related_chunk_id,
                                    entity_record["content"],
                                    entity_record["time"],
                                    entity_record["original_chunk_id"]
                                ))
                    
                    elif "Chunk" in related_labels:
                        # Direct chunk-to-chunk relationship
                        related_chunk_id = record["related_id"]
                        if related_chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(related_chunk_id)
                            chunk_result = await session.run(
                                """
                                MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                                RETURN c.content AS content, c.time AS time, c.original_chunk_id AS original_chunk_id
                                """,
                                chunk_id=related_chunk_id,
                                graph_uuid=self.neo4j_handler.run_uuid
                            )
                            async for chunk_record in chunk_result:
                                traversal_chunks.add((
                                    related_chunk_id,
                                    chunk_record["content"],
                                    chunk_record["time"],
                                    chunk_record["original_chunk_id"] if "original_chunk_id" in chunk_record.keys() else None
                                ))
                    
                    # Collect relationships
                    rel_desc = f"{record['start_name']} -[{record['rel_type']}]-> {record['end_name']}"
                    traversal_relationships.add(rel_desc)
            
            # Convert to lists
            expanded_list = [{"name": n, "batch_time": t, "source": "graph_traversal"} for n, t in expanded_entities]
            chunks_list = [{"id": cid, "content": content, "time": time, "original_chunk_id": ocid, "source": "graph_traversal"} for cid, content, time, ocid in traversal_chunks]
            relationships_list = [{"description": desc, "source": "graph_traversal"} for desc in traversal_relationships]
            
            logger.debug(f"Chunk graph expansion found {len(expanded_list)} entities, {len(chunks_list)} chunks, {len(relationships_list)} relationships")
            return expanded_list, chunks_list, relationships_list
        
        except Exception as e:
            logger.warning(f"Chunk graph expansion failed: {e}")
            return [], [], []

    async def _expand_chunks_via_pagerank(self, chunks: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Expand chunk graph using PageRank by treating chunks as starting points."""
        logger.debug("PageRank expansion from chunks not yet implemented; falling back to empty results")
        # TODO: implement if PageRank is used with chunk-seeded traversal
        return [], [], []

    async def _expand_chunks_via_ch3l3(self, chunks: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Expand chunk graph using CH3-L3 by treating chunks as starting points."""
        logger.debug("CH3-L3 expansion from chunks not yet implemented; falling back to empty results")
        # TODO: implement if CH3-L3 is used with chunk-seeded traversal
        return [], [], []

    async def _expand_entity_graph_with_chunks(self, session, entities: List[Dict], hops: int = None) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Expand entity graph based on configured hop_method.
        Dispatches to appropriate method: naive (BFS), page_rank (PPR-guided), or ch3_l3 (path-based).
        """
        hop_method = self.config.hop_method
        logger.info(f"🔍 Using hop method: {hop_method}")
        
        if hop_method == "page_rank":
            return await self._expand_via_pagerank(entities)
        elif hop_method == "ch3_l3":
            return await self._expand_via_ch3l3(entities)
        else:
            # Default: naive BFS traversal
            return await self._expand_entity_graph_naive(session, entities, hops)

    async def _expand_entity_graph_naive(self, session, entities: List[Dict], hops: int = None) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Expand entity graph via naive BFS relationship traversal, collecting entities, chunks, and relationships"""
        if not entities or not hops:
            hops = self.config.graph_hops
        
        try:
            logger.debug(f"Expanding entity graph with {hops} hops (naive BFS)")
            expanded_entities = set()
            traversal_chunks = set()
            traversal_relationships = set()
            
            for entity in entities:
                # Simple 1-hop expansion to avoid hanging on large graphs
                result = await session.run(
                    """
                    MATCH (e:Entity {name: $entity_name, graph_uuid: $graph_uuid})
                    MATCH (e)-[r]-(related)
                    WHERE related.graph_uuid = $graph_uuid
                    RETURN DISTINCT 
                        related.name AS related_name, 
                        related.id AS related_id,
                        related.batch_time AS related_batch_time,
                        labels(related) AS related_labels,
                        type(r) AS rel_type,
                        startNode(r).name AS start_name,
                        endNode(r).name AS end_name,
                        properties(r) AS rel_props
                    LIMIT 50
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
                            RETURN c.content AS content, c.time AS time, c.original_chunk_id AS original_chunk_id
                            """,
                            chunk_id=record["related_id"],
                            graph_uuid=self.neo4j_handler.run_uuid
                        )
                        async for chunk_record in chunk_result:
                            traversal_chunks.add((
                                record["related_id"],
                                chunk_record["content"],
                                chunk_record["time"],
                                chunk_record["original_chunk_id"] if "original_chunk_id" in chunk_record.keys() else None
                            ))
                    
                    # Collect relationships
                    rel_desc = f"{record['start_name']} -[{record['rel_type']}]-> {record['end_name']}"
                    traversal_relationships.add(rel_desc)
            
            # Convert to lists
            expanded_list = [{"name": n, "batch_time": t, "source": "graph_traversal"} for n, t in expanded_entities]
            chunks_list = [{"id": cid, "content": content, "time": time, "original_chunk_id": ocid, "source": "graph_traversal"} for cid, content, time, ocid in traversal_chunks]
            relationships_list = [{"description": desc, "source": "graph_traversal"} for desc in traversal_relationships]
            
            logger.debug(f"Graph expansion found {len(expanded_list)} entities, {len(chunks_list)} chunks, {len(relationships_list)} relationships")
            return expanded_list, chunks_list, relationships_list
        
        except Exception as e:
            logger.warning(f"Graph expansion failed: {e}")
            return [], [], []

    async def _expand_via_pagerank(self, entities: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Expand entity graph using precomputed Personalized PageRank scores.
        For each seed entity, retrieves top-K neighbors based on PPR.
        
        Returns:
            tuple: (expanded_entities, chunks, relationships)
        """
        if not entities:
            return [], [], []
        
        try:
            logger.debug(f"Expanding via PageRank for {len(entities)} seed entities")
            top_k = self.config.top_k_hop_pagerank
            
            expanded_entities = set()
            all_entity_names = set()
            
            # Add seed entities
            for entity in entities:
                all_entity_names.add(entity["name"])
            
            # Get PPR neighbors for each seed entity
            for entity in entities:
                ppr_neighbors = await self.neo4j_handler.get_ppr_top_neighbors(
                    entity["name"], top_k=top_k
                )
                
                for neighbor in ppr_neighbors:
                    node_name = neighbor.get("node", "")
                    if node_name:
                        expanded_entities.add(node_name)
                        all_entity_names.add(node_name)
            
            # Get chunks for all expanded entities
            chunks = await self.neo4j_handler.get_chunks_for_entities(list(all_entity_names))
            
            # Get relationships between expanded entities
            relationships = await self.neo4j_handler.get_entity_relationships(list(all_entity_names))
            
            # Convert to list format
            expanded_list = [
                {"name": n, "batch_time": "", "source": "pagerank_traversal"} 
                for n in expanded_entities
            ]
            
            logger.debug(f"PageRank expansion: {len(expanded_list)} entities, {len(chunks)} chunks, {len(relationships)} relationships")
            return expanded_list, chunks, relationships
        
        except Exception as e:
            logger.warning(f"PageRank expansion failed: {e}")
            return [], [], []

    async def _expand_via_ch3l3(self, entities: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Expand entity graph using precomputed CH3-L3 scores with path-based traversal.
        Uses beam search to find best paths from seed entities using cumulative scores.
        
        Path scoring:
        - Product aggregation: path_score = score1 * score2 * ... * scoreN
        - Sum aggregation: path_score = score1 + score2 + ... + scoreN
        
        Returns:
            tuple: (expanded_entities, chunks, relationships)
        """
        if not entities:
            return [], [], []
        
        try:
            logger.debug(f"Expanding via CH3-L3 for {len(entities)} seed entities")
            top_k_hop = self.config.top_k_hop_ch3_l3
            max_path_length = self.config.max_path_length_ch3
            aggregation = self.config.cum_score_aggregation
            
            # Track all discovered entities with their best path scores
            # Format: {entity_name: best_cumulative_score}
            entity_scores = {}
            all_entity_names = set()
            
            # Add seed entities with score 1.0
            for entity in entities:
                entity_scores[entity["name"]] = 1.0
                all_entity_names.add(entity["name"])
            
            # Beam search: explore paths up to max_path_length
            # Each beam entry: (entity_name, cumulative_score, path_length)
            beam = [(e["name"], 1.0, 0) for e in entities]
            
            for step in range(max_path_length):
                next_beam = []
                
                for current_entity, current_score, path_len in beam:
                    # Get CH3-L3 candidates for this entity
                    candidates = await self.neo4j_handler.get_ch3l3_top_candidates(
                        current_entity, top_k=top_k_hop
                    )
                    
                    for candidate in candidates:
                        node_name = candidate.get("node", "")
                        edge_score = candidate.get("score", 0.0)
                        
                        if not node_name or edge_score <= 0:
                            continue
                        
                        # Calculate cumulative score
                        if aggregation == "product":
                            new_score = current_score * edge_score
                        else:  # sum
                            new_score = current_score + edge_score
                        
                        # Update best score for this entity
                        if node_name not in entity_scores or new_score > entity_scores[node_name]:
                            entity_scores[node_name] = new_score
                            all_entity_names.add(node_name)
                        
                        # Add to next beam if we haven't reached max path length
                        if path_len + 1 < max_path_length:
                            next_beam.append((node_name, new_score, path_len + 1))
                
                # Prune beam: keep top entries per unique entity
                if next_beam:
                    # Sort by score and keep best per entity
                    next_beam.sort(key=lambda x: x[1], reverse=True)
                    seen = set()
                    pruned_beam = []
                    for entry in next_beam:
                        if entry[0] not in seen:
                            seen.add(entry[0])
                            pruned_beam.append(entry)
                    beam = pruned_beam[:50]  # Limit beam width
                else:
                    break
            
            # Sort entities by their best scores and take top ones
            sorted_entities = sorted(
                entity_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.config.top_k_entities * 3]  # Get more than top_k for diversity
            
            expanded_entity_names = [name for name, _ in sorted_entities if name not in [e["name"] for e in entities]]
            
            # Get chunks for all expanded entities
            chunks = await self.neo4j_handler.get_chunks_for_entities(list(all_entity_names))
            
            # Get relationships between expanded entities
            relationships = await self.neo4j_handler.get_entity_relationships(list(all_entity_names))
            
            # Convert to list format with scores
            expanded_list = [
                {"name": name, "batch_time": "", "score": score, "source": "ch3l3_traversal"} 
                for name, score in sorted_entities if name not in [e["name"] for e in entities]
            ]
            
            logger.debug(f"CH3-L3 expansion: {len(expanded_list)} entities, {len(chunks)} chunks, {len(relationships)} relationships")
            return expanded_list, chunks, relationships
        
        except Exception as e:
            logger.warning(f"CH3-L3 expansion failed: {e}")
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

    def _rank_expected_chunks(self, chunks: List[Dict], expected_chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Compute ranking (1-based) of expected chunk ids within an ordered chunk list."""
        ranks = []
        if not expected_chunk_ids:
            return ranks

        def _parse_id(chunk_entry: Dict[str, Any]) -> Optional[int]:
            if chunk_entry is None:
                return None
            ocid = chunk_entry.get("original_chunk_id")
            if isinstance(ocid, int):
                return ocid
            try:
                cid = str(chunk_entry.get("id", ""))
                tail = cid.split('_')[-1]
                return int(tail)
            except Exception:
                return None

        for expected in expected_chunk_ids:
            hit = {"expected_chunk_id": expected, "rank": None, "chunk_id": None, "source": None}
            for pos, chunk in enumerate(chunks, start=1):
                parsed = _parse_id(chunk)
                if parsed == expected:
                    hit["rank"] = pos
                    hit["chunk_id"] = chunk.get("id")
                    hit["source"] = chunk.get("source")
                    break
            ranks.append(hit)
        return ranks

    def _collect_question_hits(self, questions: List[Dict[str, Any]], expected_chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Summarize question nodes tied to expected chunks (no ranking available)."""
        hits = []
        if not expected_chunk_ids or not questions:
            return hits

        for q in questions:
            try:
                ocid = q.get("original_chunk_id")
                if ocid is None:
                    cid = str(q.get("chunk_id", ""))
                    ocid = int(cid.split('_')[-1]) if '_' in cid else None
            except Exception:
                ocid = None

            if ocid in expected_chunk_ids:
                hits.append({
                    "chunk_id": q.get("chunk_id"),
                    "question_id": q.get("question_id"),
                    "original_chunk_id": ocid,
                    "question": q.get("question")
                })
        return hits

    def _write_deep_analysis(self, analysis: Dict[str, Any], analysis_dir: Path, query: str) -> Optional[Path]:
        """Persist deep analysis JSON and return path."""
        try:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", query).strip("_") or "query"
            file_path = analysis_dir / f"{safe_name[:80]}.json"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            file_path.write_text(json.dumps(analysis, indent=2))
            return file_path
        except Exception as e:
            logger.warning(f"Failed to write deep analysis log: {e}")
            return None

    async def close(self):
        """Close retriever"""
        logger.info("Closing hybrid retriever")
