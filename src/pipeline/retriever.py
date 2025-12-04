import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import httpx
from ..core.logger import get_logger
from ..components.neo4j_handler import Neo4jHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # ← Changed from langchain.prompts

logger = get_logger(__name__)

class OnlineRetriever:
    """Handles online retrieval during pipeline processing"""

    def __init__(self, config, neo4j_handler: Neo4jHandler, schedule_path: str):
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
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "run_uuid": self.neo4j_handler.run_uuid,
                "executed_queries": self.executed_queries,
                "config": self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)
            }, f, indent=2)

        logger.info(f"Online retrieval results saved to: {output_path}")


class OfflineRetriever:
    """Handles offline retrieval against a specific graph UUID"""

    def __init__(self, config, neo4j_config, kg_config):
        self.config = config
        self.neo4j_config = neo4j_config
        self.kg_config = kg_config
        self.neo4j_handler = None

        # Initialize LLM for answer generation
        self.answer_llm = ChatOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            model=config.model_name,
            temperature=0.2,
        )
        
        # Answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on provided context.

Rules:
1. Answer the question directly and concisely
2. Only use information from the provided context chunks
3. If the context doesn't contain enough information, say "I cannot answer based on the available information"
4. Be specific - include details like colors, numbers, names when asked
5. Keep answers brief (1-3 sentences max)"""),
            ("user", """Question: {query}

Context from video:
{context}

Answer:""")
        ])

    async def initialize_for_graph(self, graph_uuid: str):
        """Initialize the retriever for a specific graph"""
        from ..components.neo4j_handler import Neo4jHandler
        self.neo4j_handler = Neo4jHandler(self.neo4j_config, self.kg_config, graph_uuid)

    async def retrieve(self, query: str, graph_uuid: str, groundtruth: str = "", verbose: bool = False) -> Dict[str, Any]:
        """
        Retrieve answer for a query from the knowledge graph.
        Now generates a direct answer from retrieved chunks.
        """
        start_time = time.time()
        
        try:
            # Step 1: Hybrid search for relevant chunks
            chunks = await self._hybrid_search(query, graph_uuid)
            
            # Step 2: Get related entities (optional context)
            entities = await self._get_related_entities(query, graph_uuid)
            
            # Step 3: Generate answer from chunks using LLM
            answer = await self._generate_answer(query, chunks)
            
            retrieval_time = time.time() - start_time
            
            result = {
                "query": query,
                "groundtruth": groundtruth,
                "answer": answer,  # ← Direct answer
                "graph_uuid": graph_uuid,
                "retrieval_time": retrieval_time,
            }
            
            # Add verbose info if requested
            if verbose:
                result["debug_info"] = {
                    "chunks": [
                        {"text": c["text"][:200] + "...", "score": c["score"]}
                        for c in chunks[:5]
                    ],
                    "entities": entities[:10],
                    "num_chunks_retrieved": len(chunks),
                    "num_entities_found": len(entities)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return {
                "query": query,
                "groundtruth": groundtruth,
                "answer": f"Error: {str(e)}",
                "graph_uuid": graph_uuid,
                "retrieval_time": time.time() - start_time,
            }
    
    async def _generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate a direct answer from retrieved chunks using LLM"""
        if not chunks:
            print(f"\n{'!'*50}\nNO CHUNKS FOUND FOR QUERY: {query}\n{'!'*50}\n")
            return "I cannot answer based on the available information."
        
        # Format chunks into context
        context = "\n\n".join([
            f"[Chunk {i+1}] {chunk['text']}"
            for i, chunk in enumerate(chunks[:5])  # Use top 5 chunks
        ])
        
        # VISUAL DEBUGGING
        print(f"\n{'='*50}\nQUERY: {query}\n{'-'*50}\nCONTEXT:\n{context}\n{'='*50}\n")
        
        # Generate answer using LLM
        try:
            messages = self.answer_prompt.format_messages(
                query=query,
                context=context
            )
            response = await self.answer_llm.ainvoke(messages)
            answer = response.content.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Fallback: return first chunk as answer
            return chunks[0]["text"][:300] + "..."
    
    async def _hybrid_search(self, query: str, graph_uuid: str, top_k: int = 5) -> List[Dict]:
        """Perform hybrid search (vector + keyword) on chunks"""
        neo4j_handler = Neo4jHandler(self.neo4j_config, self.kg_config, graph_uuid)
        
        try:
            vector_results = await neo4j_handler.search_chunks_vector(
                query, 
                top_k=top_k * 2,
                endpoint=self.config.endpoint,
                api_key=self.config.api_key
            )
            
            # Keyword search
            keyword_results = await neo4j_handler.search_chunks_keyword(
                query,
                top_k=top_k * 2
            )
            
            # Merge and deduplicate
            chunks_dict = {}
            
            # Add vector results
            for chunk in vector_results:
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                chunks_dict[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "score": chunk.get("score", 0),
                    "time": chunk.get("time", ""),
                    "source": "vector"
                }
            
            # Boost scores for keyword matches
            for chunk in keyword_results:
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id in chunks_dict:
                    # Keyword match found in vector results - boost score
                    chunks_dict[chunk_id]["score"] *= 1.2
                    chunks_dict[chunk_id]["source"] = "hybrid"
                else:
                    chunks_dict[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": chunk["text"],
                        "score": chunk.get("score", 0) * 0.8,  # Lower base score
                        "time": chunk.get("time", ""),
                        "source": "keyword"
                    }
            
            # Sort by score and return top k
            sorted_chunks = sorted(
                chunks_dict.values(),
                key=lambda x: x["score"],
                reverse=True
            )
            
            return sorted_chunks[:top_k]
            
        finally:
            await neo4j_handler.close()
    
    async def _get_related_entities(self, query: str, graph_uuid: str, limit: int = 20) -> List[str]:
        """Get entity names related to the query"""
        neo4j_handler = Neo4jHandler(self.neo4j_config, self.kg_config, graph_uuid)
        
        try:
            # Simple keyword matching on entity names
            query_keywords = query.lower().split()
            
            cypher = """
            MATCH (e:Entity)
            WHERE e.run_uuid = $run_uuid
            AND any(keyword IN $keywords WHERE toLower(e.name) CONTAINS keyword)
            RETURN e.name as name
            LIMIT $limit
            """
            
            async with neo4j_handler.driver.session() as session:
                result = await session.run(
                    cypher,
                    {
                        "run_uuid": graph_uuid,
                        "keywords": query_keywords,
                        "limit": limit
                    }
                )
                records = await result.values()
                return [record[0] for record in records if record[0]]
                
        except Exception as e:
            logger.warning(f"Entity retrieval failed: {e}")
            return []
        finally:
            await neo4j_handler.close()
    
    async def batch_retrieve_from_file(self, input_file: str, graph_uuid: str, verbose: bool = False) -> List[Dict]:
        """Process multiple queries from a JSON file"""
        logger.info(f"Loading queries from: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        results = []
        for i, query_item in enumerate(queries, 1):
            query = query_item["query"]
            groundtruth = query_item.get("groundtruth", "")
            
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            
            result = await self.retrieve(query, graph_uuid, groundtruth, verbose)
            results.append(result)
        
        return results