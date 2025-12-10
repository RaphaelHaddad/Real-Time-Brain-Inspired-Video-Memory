"""
Edge Retrieval: Vector search on relationship descriptions (Edge Sentences).
Implements: 1) Entity search, 2) Relationship expansion, 3) Edge-to-Query Similarity
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.config import RetrievalConfig
from ..components.neo4j_handler import Neo4jHandler
from ..core.logger import get_logger
from langchain_openai import OpenAIEmbeddings

logger = get_logger(__name__)

class EdgeRetriever:
    """
    Retrieves information by scoring 'Edge Sentences' against the query.
    Edge Sentence format: "{start_node} {rel_type} {end_node}"
    """

    def __init__(self, config: RetrievalConfig, neo4j_handler: Neo4jHandler, schedule_path: Optional[str], realtime_output: bool = True):
        self.config = config
        self.neo4j_handler = neo4j_handler
        self.schedule_path = schedule_path
        self.retrieval_schedule = self._load_retrieval_schedule() if schedule_path else []
        self.executed_queries = []
        self.executed_query_keys = set()
        self.output_file = Path("edge_retrieval_results.json") if realtime_output else None
        if self.output_file and realtime_output:
            self.output_file.write_text("[]\n")
        
        # Initialize embedder for edge scoring
        self.embedder = OpenAIEmbeddings(
            model=neo4j_handler.kg_config.embedding_model,
            openai_api_key=neo4j_handler.kg_config.embedding_api_key,
            openai_api_base=neo4j_handler.kg_config.embedding_endpoint
        )
        
        logger.info(f"Initialized EdgeRetriever: entities_top_k={config.top_k_entities}")

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
        """Check and execute scheduled queries at current time"""
        logger.info(f"Checking for queries at batch end time: {current_video_time}")
        results = []

        for query_schedule in self.retrieval_schedule:
            scheduled_time = query_schedule.get('time', '')
            
            # Handle time ranges
            if '-' in current_video_time:
                end_time = current_video_time.split('-')[1].strip()
                time_matches = scheduled_time <= end_time
            else:
                end_time = current_video_time
                time_matches = scheduled_time == end_time
            
            if time_matches:
                query = query_schedule.get('query', '')
                groundtruth = query_schedule.get('groundtruth', '')
                
                # Skip if this (query, scheduled_time) was already executed
                query_key = (query, scheduled_time)
                if query_key in self.executed_query_keys:
                    continue
                
                logger.info(f"🔍 EDGE RETRIEVAL TRIGGERED at {end_time}")
                logger.info(f"   Query: {query}")
                
                query_start = time.perf_counter()
                try:
                    retrieval_result = await self._perform_edge_retrieval(query)
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
                    self.executed_query_keys.add(query_key)
                    self._write_result_realtime(result)
                    
                    logger.info(f"✓ Edge Retrieval completed in {query_time:.3f}s")
                    logger.info(f"   Result: {str(retrieval_result)[:100]}...")
                
                except Exception as e:
                    logger.error(f"Error in edge retrieval: {str(e)}")

        return results

    async def _perform_edge_retrieval(self, query: str) -> str:
        """
        Perform edge retrieval:
        1. Fulltext search for entities.
        2. Expand to get relationships.
        3. Form edge sentences.
        4. Embed and rank edges against query.
        """
        try:
            logger.debug(f"Starting edge retrieval for query: '{query}'")
            retrieval_start = time.perf_counter()
            
            # 1. Get Query Embedding
            query_embedding = await self.embedder.aembed_query(query)
            
            async with self.neo4j_handler.driver.session() as session:
                # 2. Find Candidate Entities
                entity_results = await self._fulltext_search_entities(session, query)
                if not entity_results:
                    return f"No entities found for query '{query}'"

                # 3. Expand to get Relationships (Edges)
                edges = await self._get_candidate_edges(session, entity_results)
                
                if not edges:
                    return f"No relationships found connected to entities for '{query}'"

                # 4. Form Sentences and Embed
                # Deduplicate edges based on ID
                unique_edges = {e['id']: e for e in edges}.values()
                
                edge_sentences = []
                edge_objects = []
                
                for edge in unique_edges:
                    # Format: "NodeA RELATION_TYPE NodeB"
                    rel_type_clean = edge['type'].replace('_', ' ')
                    sentence = f"{edge['start']} {rel_type_clean} {edge['end']}"
                    edge_sentences.append(sentence)
                    edge_objects.append({
                        **edge,
                        "sentence": sentence
                    })
                
                logger.debug(f"Embedding {len(edge_sentences)} edge sentences...")
                # Batch embedding
                edge_embeddings = await self.embedder.aembed_documents(edge_sentences)
                
                # 5. Calculate Similarity
                scored_edges = []
                for i, edge_emb in enumerate(edge_embeddings):
                    score = self._cosine_similarity(query_embedding, edge_emb)
                    scored_edges.append({
                        **edge_objects[i],
                        "score": score
                    })
                
                # 6. Sort and Format
                scored_edges.sort(key=lambda x: x['score'], reverse=True)
                # Use top_k_relationships from config, default to 10 if not set
                top_k = getattr(self.config, 'top_k_relationships', 10)
                top_edges = scored_edges[:top_k]
                
                result_text = self._format_edge_results(query, top_edges)
                
                total_time = time.perf_counter() - retrieval_start
                logger.debug(f"Total edge retrieval time: {total_time:.3f}s")
                return result_text

        except Exception as e:
            logger.error(f"Edge retrieval error: {str(e)}")
            return f"Edge retrieval failed: {str(e)}"

    async def _fulltext_search_entities(self, session, query: str) -> List[Dict[str, Any]]:
        """Fulltext search on entity names"""
        try:
            result = await session.run(
                """
                CALL db.index.fulltext.queryNodes("entityName", $search_query)
                YIELD node, score
                WHERE node.graph_uuid = $graph_uuid
                RETURN node.name AS name, elementId(node) as id, score
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
                    "id": record["id"],
                    "score": float(record["score"])
                })
            return entities
        except Exception as e:
            logger.warning(f"Fulltext search failed: {e}")
            return []

    async def _get_candidate_edges(self, session, entities: List[Dict]) -> List[Dict]:
        """Get relationships connected to the candidate entities."""
        edges = []
        entity_names = [e['name'] for e in entities]
        
        try:
            # Fetch 1-hop relationships
            result = await session.run(
                """
                MATCH (start:Entity {graph_uuid: $graph_uuid})-[r]-(end:Entity {graph_uuid: $graph_uuid})
                WHERE start.name IN $names
                RETURN 
                    elementId(r) as id,
                    start.name as start,
                    type(r) as type,
                    end.name as end,
                    r.source_chunks as source_chunks
                LIMIT 200
                """,
                names=entity_names,
                graph_uuid=self.neo4j_handler.run_uuid
            )
            
            async for record in result:
                edges.append({
                    "id": record["id"],
                    "start": record["start"],
                    "type": record["type"],
                    "end": record["end"],
                    "source_chunks": record["source_chunks"]
                })
            return edges
        except Exception as e:
            logger.warning(f"Edge fetch failed: {e}")
            return []

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 * mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def _format_edge_results(self, query: str, edges: List[Dict]) -> str:
        if not edges:
            return f"No relevant edges found for '{query}'"
        
        parts = [f"Top Edges for '{query}':"]
        for i, edge in enumerate(edges, 1):
            parts.append(f"{i}. {edge['sentence']} (score: {edge['score']:.3f})")
        
        return "\n".join(parts)

    async def close(self):
        """Close retriever"""
        logger.info("Closing EdgeRetriever")