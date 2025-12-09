import json
import time
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.config import RetrievalConfig
from ..components.neo4j_handler import Neo4jHandler
from ..core.logger import get_logger
from .retriever_hybrid import HybridRetriever, RerankerError

logger = get_logger(__name__)

class OnlineRetriever:
    """Handles online retrieval during pipeline processing"""

    def __init__(self, config: RetrievalConfig, neo4j_handler: Neo4jHandler, schedule_path: str):
        self.config = config
        self.neo4j_handler = neo4j_handler
        self.schedule_path = schedule_path
        self.retrieval_schedule = self._load_retrieval_schedule()
        self.executed_queries = []
        self.output_file = Path("retrieval_results.json")
        # Initialize empty results file
        self.output_file.write_text("[]\n")

    def _write_result_realtime(self, result: Dict):
        """Write a single retrieval result to file in real-time"""
        try:
            # Read existing results
            existing = json.loads(self.output_file.read_text()) if self.output_file.exists() else []
            # Append new result
            existing.append(result)
            # Write back
            self.output_file.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.error(f"Failed to write real-time result: {e}")

    def _load_retrieval_schedule(self) -> List[Dict[str, str]]:
        """Load the retrieval schedule from a JSON file"""
        try:
            with open(self.schedule_path, 'r') as f:
                schedule_data = json.load(f)
                # Validate the structure
                if 'queries' in schedule_data:
                    return schedule_data['queries']
                else:
                    # Assume it's a list of queries
                    return schedule_data
        except Exception as e:
            logger.error(f"Error loading retrieval schedule: {str(e)}")
            return []

    async def check_and_run_queries(self, current_video_time: str) -> List[Dict]:
        """Check if there are queries scheduled for the current time and execute them"""
        logger.info(f"Checking for queries at batch end time: {current_video_time}")
        results = []

        for query_schedule in self.retrieval_schedule:
            scheduled_time = query_schedule.get('time', '')
            
            # Handle time ranges in VLM output (e.g., "00:14-00:19")
            # Extract the end time from the range for comparison
            if '-' in current_video_time:
                end_time = current_video_time.split('-')[1].strip()
            else:
                end_time = current_video_time
            
            logger.debug(f"Comparing scheduled_time '{scheduled_time}' with batch_end_time '{end_time}'")

            # Check if this query should be executed at the current time
            # Match if scheduled time is <= batch end time and > batch start time (within the range)
            if '-' in current_video_time:
                start_time = current_video_time.split('-')[0].strip()
                # Query triggers if scheduled_time is within or before the batch end
                time_matches = scheduled_time <= end_time
            else:
                start_time = current_video_time
                time_matches = scheduled_time == end_time
            
            if time_matches:
                logger.info(f"🔍 RETRIEVAL TRIGGERED at batch end time {end_time} (range: {current_video_time})")
                logger.info(f"   Query: {query_schedule.get('query', '')}")
                query_start = time.perf_counter()

                try:
                    # Support both old and new format for compatibility
                    query = query_schedule.get('query') or query_schedule.get('question', '')
                    groundtruth = query_schedule.get('groundtruth') or query_schedule.get('answer', '')

                    # Perform the retrieval
                    retrieval_result = await self._perform_retrieval(query)

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

                    # Write result in real-time
                    self._write_result_realtime(result)

                    logger.info(f"✓ Online retrieval completed in {query_time:.2f}s")
                    logger.info(f"   Result: {str(retrieval_result)[:100]}...")

                except Exception as e:
                    logger.error(f"Error executing online retrieval: {str(e)}")

        return results

    async def _perform_retrieval(self, query: str) -> str:
        """Perform a retrieval query against the Neo4j database"""
        try:
            logger.debug(f"Starting retrieval for query: '{query}'")
            
            # This is a placeholder - in a real implementation, you'd perform
            # actual graph queries based on the query string
            async with self.neo4j_handler.driver.session() as session:
                # Basic example: find entities related to the query
                logger.debug(f"Executing fulltext search with query: '{query}'")
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes("entityName", $search_query)
                    YIELD node, score
                    RETURN node.name AS name, node.graph_uuid AS graph_uuid, node.batch_time AS batch_time, score
                    LIMIT $limit_count
                    """,
                    search_query=query,
                    limit_count=self.config.top_k * 2
                )

                nodes = []
                async for record in result:
                    nodes.append({
                        "name": record["name"],
                        "graph_uuid": record["graph_uuid"],
                        "batch_time": record["batch_time"] or "",
                        "score": record["score"]
                    })

                logger.debug(f"Fulltext search returned {len(nodes)} raw nodes: {[n['name'] for n in nodes[:5]]}{'...' if len(nodes) > 5 else ''}")

                # If reranker is configured, apply reranking
                if self.config.use_reranker and self.config.reranker_endpoint and self.config.reranker_model:
                    logger.debug("Applying reranking to results")
                    nodes = await self._rerank_results_async(query, nodes)
                    logger.debug(f"After reranking: {len(nodes)} nodes: {[n['name'] for n in nodes[:5]]}{'...' if len(nodes) > 5 else ''}")

                # Limit to top_k results after potential reranking
                nodes = nodes[:self.config.top_k]
                logger.debug(f"After limiting to top_k ({self.config.top_k}): {len(nodes)} final nodes")

                # Format results with time information
                result_parts = []
                for node in nodes:
                    time_info = f" (time: {node['batch_time']})" if node['batch_time'] else ""
                    result_parts.append(f"- {node['name']}{time_info} (score: {node['score']:.3f})")
                
                result_text = f"Found {len(nodes)} nodes related to query '{query[:50]}...':\n" + "\n".join(result_parts)
                logger.debug(f"Final retrieval result: {result_text}")
                return result_text

        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return f"Retrieval failed: {str(e)}"

    async def _rerank_results_async(self, query: str, nodes: List[Dict]) -> List[Dict]:
        """Async reranking for online retriever"""
        try:
            if not nodes:
                logger.debug("No nodes to rerank, returning empty list")
                return nodes

            logger.debug(f"Reranking {len(nodes)} nodes for query: '{query}'")
            
            # Prepare documents for reranking
            documents = [node["name"] for node in nodes]
            logger.debug(f"Documents for reranking: {documents}")

            # Prepare the request to the reranker API
            rerank_payload = {
                "query": query,
                "documents": documents,
                "top_k": len(documents)  # Rerank all documents
            }

            # Add authentication if reranker API key is provided
            headers = {"Content-Type": "application/json"}
            if self.config.reranker_api_key:
                headers["Authorization"] = f"Bearer {self.config.reranker_api_key}"

            logger.debug(f"Sending rerank request to {self.config.reranker_endpoint}")
            
            # Make the API call to the reranker
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.config.reranker_endpoint,
                    json=rerank_payload,
                    headers=headers
                )
                response.raise_for_status()
                
                rerank_result = response.json()
                logger.debug(f"Reranker response: {rerank_result}")

            # Reorder nodes based on rerank result
            # The reranker should return indices or scores for reranking
            if "results" in rerank_result and isinstance(rerank_result["results"], list):
                # Assuming rerank_result["results"] is a list of {index, score} objects
                reranked_indices = [item["index"] for item in rerank_result["results"]]
                reranked_nodes = [nodes[i] for i in reranked_indices if i < len(nodes)]
                logger.debug(f"Reranked node order: {[nodes[i]['name'] for i in reranked_indices if i < len(nodes)]}")
                return reranked_nodes
            else:
                logger.warning("Reranker response format not as expected, returning original order")
                return nodes

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.info("Returning original results without reranking")
            return nodes

    async def close(self):
        """Close the online retriever and save execution results"""
        # Save retrieval results to a file
        output_path = f"outputs/retrieval_results_{self.neo4j_handler.run_uuid}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "run_uuid": self.neo4j_handler.run_uuid,
                "executed_queries": self.executed_queries,
                "config": self.config.dict()
            }, f, indent=2)

        logger.info(f"Online retrieval results saved to: {output_path}")


class OfflineRetriever:
    """Handles offline retrieval against a specific graph UUID"""

    def __init__(self, config: RetrievalConfig, neo4j_config, kg_config, community_config=None):
        self.config = config
        self.neo4j_config = neo4j_config
        self.kg_config = kg_config
        self.community_config = community_config
        self.neo4j_handler = None

    async def initialize_for_graph(self, graph_uuid: str):
        """Initialize the retriever for a specific graph"""
        from ..components.neo4j_handler import Neo4jHandler
        self.neo4j_handler = Neo4jHandler(self.neo4j_config, self.kg_config, graph_uuid)

    async def retrieve(self, query: str, graph_uuid: str, groundtruth: str = "", true_chunks: List[int] = None, expected_chunk_ids: List[int] = None, analysis_dir: Path = None) -> Dict[str, Any]:
        """Perform offline retrieval against the specified graph"""
        await self.initialize_for_graph(graph_uuid)

        start_time = time.perf_counter()
        try:
            # Perform the retrieval and pass true_chunks info
            retrieval_result, reranking_performed, analysis_path = await self._perform_retrieval(
                query, true_chunks, expected_chunk_ids, analysis_dir
            )
            retrieval_time = time.perf_counter() - start_time

            if reranking_performed:
                print("Reranking successful")

            result = {
                "query": query,
                "groundtruth": groundtruth,
                "retrieval": retrieval_result,
                "graph_uuid": graph_uuid,
                "retrieval_time": retrieval_time,
                "verbose": self.config.verbose
            }

            if analysis_path:
                result["analysis_log"] = str(analysis_path)

            if self.config.verbose:
                logger.info(f"Offline retrieval details:")
                logger.info(f"  Query: {query}")
                logger.info(f"  Ground truth: {groundtruth}")
                logger.info(f"  Graph UUID: {graph_uuid}")
                logger.info(f"  Time: {retrieval_time:.2f}s")
                logger.info(f"  Retrieval result: {retrieval_result}")
                if analysis_path:
                    logger.info(f"  Deep analysis log: {analysis_path}")

            return result

        except RerankerError:
            # For offline retrieval, a RerankerError should bubble up and abort the batch
            logger.error("Reranker failed (strict mode). Aborting retrieval and propagating error.")
            raise
        except Exception as e:
            logger.error(f"Error in offline retrieval: {str(e)}")
            return {
                "query": query,
                "groundtruth": groundtruth,
                "graph_uuid": graph_uuid,
                "retrieval": f"Error: {str(e)}",
                "retrieval_time": time.perf_counter() - start_time
            }
        finally:
            if self.neo4j_handler:
                await self.neo4j_handler.close()

    async def batch_retrieve_from_file(self, input_file_path: str, graph_uuid: str, expected_chunk_json: str = None, deep_analysis_dir: str = None) -> List[Dict[str, Any]]:
        """Perform batch offline retrieval from a JSON file with consistent format"""
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                queries_data = json.load(f)

            # Load expected chunk mapping (query -> list[int]) if provided
            expected_map = {}
            analysis_dir = None
            if expected_chunk_json:
                try:
                    with open(expected_chunk_json, 'r', encoding='utf-8') as ef:
                        expected_payload = json.load(ef)
                        items = expected_payload.get('retrieval_analysis') or []
                        for item in items:
                            q = item.get('query')
                            if not q:
                                continue
                            ids = item.get('expected_chunk_ids') or []
                            try:
                                expected_map[q] = [int(x) for x in ids]
                            except Exception:
                                expected_map[q] = []

                    timestamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
                    # If caller provided a deep_analysis_dir, use that as base, else default to logs/deep_retrieval
                    if deep_analysis_dir:
                        analysis_dir = Path(deep_analysis_dir) / graph_uuid / timestamp
                    else:
                        analysis_dir = Path('logs') / 'deep_retrieval' / graph_uuid / timestamp
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to load expected_chunk_json '{expected_chunk_json}': {e}")
                    expected_map = {}
                    analysis_dir = None
            
            results = []
            for item in queries_data:
                query = item.get('query', '')
                groundtruth = item.get('groundtruth', '')
                # Allow item to provide optional true_chunks array
                true_chunks = item.get('true_chunks') or item.get('true_chunk')
                parsed_true_chunks = None
                if true_chunks:
                    try:
                        if isinstance(true_chunks, list):
                            parsed_true_chunks = [int(x) for x in true_chunks]
                        elif isinstance(true_chunks, str):
                            parts = [p.strip() for p in true_chunks.strip('[]').split(',') if p.strip()]
                            parsed_true_chunks = [int(x) for x in parts]
                    except Exception:
                        parsed_true_chunks = None

                expected_ids = expected_map.get(query)
                result = await self.retrieve(query, graph_uuid, groundtruth, parsed_true_chunks, expected_ids, analysis_dir)
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error in batch offline retrieval: {str(e)}")
            return []

    async def _perform_retrieval(self, query: str, true_chunks: List[int] = None, expected_chunk_ids: List[int] = None, analysis_dir: Path = None) -> tuple[str, bool, Optional[Path]]:
        """Perform a retrieval query using the hybrid search logic, with optional true_chunks tracking"""
        hybrid = HybridRetriever(
            self.config, 
            self.neo4j_handler, 
            schedule_path=None, 
            realtime_output=False,
            community_config=self.community_config
        )
        return await hybrid._perform_hybrid_retrieval(query, true_chunks, expected_chunk_ids, analysis_dir)

    async def _rerank_results(self, query: str, nodes: List[Dict]) -> List[Dict]:
        """Apply reranking to the retrieval results if a reranker is configured"""
        try:
            if not nodes:
                return nodes

            # Prepare documents for reranking
            documents = [node["name"] for node in nodes]

            # Prepare the request to the reranker API
            rerank_payload = {
                "query": query,
                "documents": documents,
                "top_k": len(documents)  # Rerank all documents
            }

            # Add authentication if reranker API key is provided
            headers = {"Content-Type": "application/json"}
            if self.config.reranker_api_key:
                headers["Authorization"] = f"Bearer {self.config.reranker_api_key}"

            # Make the API call to the reranker
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.config.reranker_endpoint,
                    json=rerank_payload,
                    headers=headers
                )
                response.raise_for_status()
                
                rerank_result = response.json()

            # Reorder nodes based on rerank result
            # The reranker should return indices or scores for reranking
            if "results" in rerank_result and isinstance(rerank_result["results"], list):
                # Assuming rerank_result["results"] is a list of {index, score} objects
                reranked_indices = [item["index"] for item in rerank_result["results"]]
                reranked_nodes = [nodes[i] for i in reranked_indices if i < len(nodes)]
                return reranked_nodes
            else:
                logger.warning("Reranker response format not as expected, returning original order")
                return nodes

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.info("Returning original results without reranking")
            return nodes