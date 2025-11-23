"""
Pre-LLM Injector: Hierarchical chunking & local triplet extraction
Uses TokenTextSplitter + LLMGraphTransformer for efficient local extraction
"""

import asyncio
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from .prompts import build_pre_llm_prompt_template
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from ..core.config import ChunkingConfig, EmbedderConfig, LLMInjectorConfig
from ..core.logger import get_logger

logger = get_logger(__name__)


class PreLLMInjector:
    """Local extraction from chunks using optimized pipe-delimited LLM calls"""

    def __init__(self, llm: BaseChatModel, config: ChunkingConfig, embedder_config: EmbedderConfig = None, llm_injector_config: LLMInjectorConfig = None):
        self.config = config
        self.llm = llm
        self.embedder_config = embedder_config
        self.llm_injector_config = llm_injector_config
        # Holds details from the last extraction for external tracing/logging
        self.last_chunk_details: List[Dict[str, Any]] = []
        # Build a ChatPromptTemplate using the centralized prompts module so prompts
        # can be edited in one place. The helper will inject the configured
        # max_triplets while preserving the {input} placeholder.
        self.prompt_template = build_pre_llm_prompt_template(config.max_triplets_per_chunk)

        # Create a simple chain for direct LLM calls
        self.chain = self.llm

        # Initialize text splitter with overlap for context preservation
        self.splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        logger.info(
            f"Initialized PreLLMInjector: chunk_size={config.chunk_size}, "
            f"overlap={config.chunk_overlap}, parallel={config.parallel_count}, "
            f"max_triplets_per_chunk={config.max_triplets_per_chunk}, "
            f"subgraph_extraction={llm_injector_config.subgraph_extraction_injection if llm_injector_config else False}"
        )

    async def extract_local_triplets(
        self, content: str, network_info: str = "", neo4j_handler = None
    ) -> List[Dict[str, Any]]:
        """
        Extract triplets from content using hierarchical chunking + local extraction

        Args:
            content: Aggregated VLM output text
            network_info: Optional graph context (unused in local extraction)

        Returns:
            List of triplet dicts with 'head', 'relation', 'tail', 'source_chunks'
        """
        try:
            # Split content into manageable chunks
            chunks = self.splitter.split_text(content)
            logger.debug(
                f"Split content into {len(chunks)} chunks (chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
            )

            # Log full content of chunks to aid debugging
            for i, c in enumerate(chunks):
                logger.debug(f"Chunk[{i}] ~{len(c.split())} words, full content: {c}")

            if not chunks:
                logger.warning("No chunks produced from content")
                return []

            # Extract triplets from each chunk in parallel
            if self.llm_injector_config and self.llm_injector_config.subgraph_extraction_injection:
                triplets = await self._parallel_chunk_extraction_with_similarity(chunks, neo4j_handler)
            else:
                triplets = await self._parallel_chunk_extraction(chunks)

            # Deduplicate at local level (preserves source_chunks)
            triplets = self._deduplicate_triplets(triplets)

            logger.info(
                f"Extracted {len(triplets)} local triplets from {len(chunks)} chunks"
            )
            return triplets

        except Exception as e:
            logger.error(f"Pre-LLM extraction failed: {str(e)}")
            return []

    async def _parallel_chunk_extraction(
        self, chunks: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract triplets from multiple chunks in parallel or serially based on config, tracking chunk indices"""
        if not self.config.batch_llm_parallelism:
            # Serial execution
            logger.debug("Using serial LLM extraction (batch_llm_parallelism=false)")
            all_triplets: List[Dict[str, Any]] = []
            self.last_chunk_details = []
            for i, chunk in enumerate(chunks):
                try:
                    result = await self._extract_chunk_triplets(chunk, i)
                    logger.debug(f"Chunk {i} returned {len(result)} triplets")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "triplets": result,
                    })
                    all_triplets.extend(result)
                except Exception as e:
                    logger.warning(f"Chunk {i} extraction failed: {e}")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "triplets": [],
                        "error": str(e),
                    })
            return all_triplets
        
        # Parallel execution (original logic)
        # Limit concurrency to avoid overwhelming the LLM
        semaphore = asyncio.Semaphore(self.config.parallel_count)

        async def extract_with_semaphore(chunk_idx: int, chunk: str) -> List[Dict[str, Any]]:
            async with semaphore:
                return await self._extract_chunk_triplets(chunk, chunk_idx)

        # Launch parallel tasks with chunk indices
        tasks = [extract_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter errors
        all_triplets: List[Dict[str, Any]] = []
        # Reset last chunk details for this run
        self.last_chunk_details = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Chunk {i} extraction failed: {result}")
                # Record chunk text even if failed
                self.last_chunk_details.append({
                    "chunk_index": i,
                    "chunk_text": chunks[i],
                    "triplets": [],
                    "error": str(result),
                })
            else:
                logger.debug(f"Chunk {i} returned {len(result)} triplets")
                self.last_chunk_details.append({
                    "chunk_index": i,
                    "chunk_text": chunks[i],
                    "triplets": result,
                })
                all_triplets.extend(result)

        return all_triplets

    async def _parallel_chunk_extraction_with_similarity(self, chunks: List[str], neo4j_handler) -> List[Dict[str, Any]]:
        """Extract triplets with subgraph similarity calculation"""
        import asyncio
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize embedder
        embedder = OpenAIEmbeddings(
            base_url=self.embedder_config.endpoint,
            api_key=self.embedder_config.api_key,
            model=self.embedder_config.model,
        )
        
        # Start embedding tasks for all chunks (keep parallel as embeddings are fast)
        embedding_tasks = []
        for chunk in chunks:
            task = embedder.aembed_query(chunk)
            embedding_tasks.append(task)
        
        # Wait for embeddings to complete
        logger.debug("Starting chunk embedding tasks...")
        chunk_embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
        
        # Calculate similarities against existing batch nodes
        if neo4j_handler:
            batch_similarities = await self._calculate_batch_similarities(chunk_embeddings, neo4j_handler)
            
            # Collect all batch similarities across chunks
            all_batch_scores = {}
            for chunk_similarities in batch_similarities:
                for batch_id, score in chunk_similarities:
                    if batch_id not in all_batch_scores:
                        all_batch_scores[batch_id] = []
                    all_batch_scores[batch_id].append(score)
            
            # Average scores per batch and sort
            avg_batch_scores = [(bid, sum(scores)/len(scores)) for bid, scores in all_batch_scores.items()]
            avg_batch_scores.sort(key=lambda x: x[1], reverse=True)
            top_similar_batches = avg_batch_scores[:self.embedder_config.top_k_similar_batch]
            
            logger.debug(f"Overall top similar batches: {[(bid, f'{score:.3f}') for bid, score in top_similar_batches]}")
            
            # Log per-chunk similarities
            for chunk_idx, similarities in enumerate(batch_similarities):
                if similarities:
                    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:self.embedder_config.top_k_similar_batch]
                    logger.debug(f"Chunk {chunk_idx} top similar batches: {[(bid, f'{score:.3f}') for bid, score in top_similar]}")
                else:
                    logger.debug(f"Chunk {chunk_idx} top similar batches: none (no existing batches to compare)")
        else:
            logger.debug("Skipping batch similarity calculation (no neo4j_handler)")
        
        # LLM extraction: serial or parallel based on config
        if not self.config.batch_llm_parallelism:
            # Serial execution
            logger.debug("Using serial LLM extraction with similarity (batch_llm_parallelism=false)")
            all_triplets: List[Dict[str, Any]] = []
            self.last_chunk_details = []
            for i, chunk in enumerate(chunks):
                try:
                    result = await self._extract_chunk_triplets(chunk, i)
                    logger.debug(f"Chunk {i} returned {len(result)} triplets")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "triplets": result,
                    })
                    all_triplets.extend(result)
                except Exception as e:
                    logger.warning(f"Chunk {i} extraction failed: {e}")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "triplets": [],
                        "error": str(e),
                    })
            return all_triplets
        
        # Parallel LLM extraction (original logic)
        semaphore = asyncio.Semaphore(self.config.parallel_count)
        async def extract_with_semaphore(chunk_idx: int, chunk: str) -> List[Dict[str, Any]]:
            async with semaphore:
                return await self._extract_chunk_triplets(chunk, chunk_idx)
        
        llm_tasks = [extract_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]
        
        # Wait for LLM extractions to complete
        logger.debug("Waiting for LLM extraction tasks...")
        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        
        # Process results (same as original method)
        all_triplets: List[Dict[str, Any]] = []
        self.last_chunk_details = []
        for i, result in enumerate(llm_results):
            if isinstance(result, Exception):
                logger.warning(f"Chunk {i} extraction failed: {result}")
                self.last_chunk_details.append({
                    "chunk_index": i,
                    "chunk_text": chunks[i],
                    "triplets": [],
                    "error": str(result),
                })
            else:
                logger.debug(f"Chunk {i} returned {len(result)} triplets")
                self.last_chunk_details.append({
                    "chunk_index": i,
                    "chunk_text": chunks[i],
                    "triplets": result,
                })
                all_triplets.extend(result)

        return all_triplets

    async def _calculate_batch_similarities(self, chunk_embeddings, neo4j_handler) -> List[List[tuple[int, float]]]:
        """Calculate similarity between chunks and existing batch nodes"""
        import math
        
        # Query existing batch nodes and their embeddings
        batch_embeddings = await self._get_batch_embeddings(neo4j_handler)
        
        similarities = []
        for chunk_emb in chunk_embeddings:
            if isinstance(chunk_emb, Exception):
                similarities.append([])
                continue
            
            chunk_similarities = []
            for batch_id, batch_emb in batch_embeddings.items():
                if batch_emb:
                    # Calculate cosine similarity manually
                    similarity = self._cosine_similarity(chunk_emb, batch_emb)
                    chunk_similarities.append((batch_id, float(similarity)))
            
            # Sort by similarity and take top_k
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            similarities.append(chunk_similarities[:self.embedder_config.top_k_chunk_with_batch_similarity])
        
        return similarities

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    async def _get_batch_embeddings(self, neo4j_handler) -> Dict[int, List[float]]:
        """Get embeddings for existing batch chunks in current graph"""
        try:
            async with neo4j_handler.driver.session() as session:
                result = await session.run("""
                    MATCH (c:Chunk:GraphNode)
                    WHERE c.graph_uuid = $graph_uuid AND c.batch_id IS NOT NULL AND c.embedding IS NOT NULL
                    RETURN c.batch_id as batch_id, c.embedding as embedding
                    LIMIT 1000
                """, graph_uuid=neo4j_handler.run_uuid)
                
                batch_embeddings = {}
                async for record in result:
                    batch_id = record["batch_id"]
                    embedding = record["embedding"]
                    if isinstance(embedding, list):
                        # Average embeddings per batch (or take first, or some aggregation)
                        if batch_id not in batch_embeddings:
                            batch_embeddings[batch_id] = embedding
                        else:
                            # Simple average for now
                            current = batch_embeddings[batch_id]
                            batch_embeddings[batch_id] = [(a + b) / 2 for a, b in zip(current, embedding)]
                
                return batch_embeddings
        except Exception as e:
            logger.warning(f"Failed to get batch embeddings: {e}")
            return {}

    async def _extract_chunk_triplets(self, chunk_text: str, chunk_idx: int) -> List[Dict]:
        """
        Extracts triplets from a single chunk using the LLM with a STRICT TIMEOUT.
        """
        prompt = self.prompt_template.invoke({"input": chunk_text})
        
        # TIMEOUT CONFIG: 
        # For local LLMs, 45-60s is usually safe. 
        # If it takes longer, the server is likely stuck.
        TIMEOUT_SECONDS = 45.0 

        try:
            logger.debug(f"Calling LLM on chunk {chunk_idx} (~{len(chunk_text.split())} words)")
            
            # THE FIX: Wrap the API call in wait_for
            response = await asyncio.wait_for(
                self.chain.ainvoke(prompt),
                timeout=TIMEOUT_SECONDS
            )

            # Parse the pipe-delimited output
            llm_output = getattr(response, 'content', str(response)).strip()
            triplets = self._parse_pipe_delimited_output(llm_output, chunk_idx)
            logger.debug(f"Extracted {len(triplets)} triplets from chunk {chunk_idx}")
            return triplets

        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Chunk {chunk_idx} TIMED OUT after {TIMEOUT_SECONDS}s. Skipping.")
            return [] # Return empty list so pipeline continues
            
        except Exception as e:
            logger.error(f"Error extracting from chunk {chunk_idx}: {e}")
            return []

    def _parse_pipe_delimited_output(self, llm_output: str, chunk_idx: int) -> List[Dict[str, Any]]:
        """Parse pipe-delimited triplet output into dict format"""
        triplets = []
        lines = llm_output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            parts = line.split('|')
            if len(parts) == 3:
                head = parts[0].strip()
                relation = parts[1].strip()
                tail = parts[2].strip()
                
                if head and relation and tail:
                    triplet = {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "source_chunks": [chunk_idx]
                    }
                    triplets.append(triplet)
                    
                    # Respect max_triplets_per_chunk limit
                    if len(triplets) >= self.config.max_triplets_per_chunk:
                        break
        
        return triplets

    def _deduplicate_triplets(
        self, triplets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate triplets by (head, relation, tail), merging source_chunks"""
        seen = {}  # Maps key → first occurrence with merged chunks
        deduped = []
        
        for triplet in triplets:
            # Defensive handling: if transformer returned a JSON string or unexpected type
            if isinstance(triplet, str):
                try:
                    parsed = json.loads(triplet)
                    if isinstance(parsed, dict):
                        triplet = parsed
                    else:
                        logger.debug(f"Ignoring non-dict parsed triplet: {parsed}")
                        continue
                except Exception:
                    logger.debug(f"Ignoring non-json triplet string: {triplet}")
                    continue

            if not isinstance(triplet, dict):
                logger.debug(f"Ignoring triplet with unexpected type: {type(triplet)}")
                continue

            head = str(triplet.get("head", "")).strip().lower()
            relation = str(triplet.get("relation", "")).strip().lower()
            tail = str(triplet.get("tail", "")).strip().lower()
            source_chunks = triplet.get("source_chunks", [])
            
            # Ensure source_chunks is a list
            if not isinstance(source_chunks, list):
                source_chunks = [source_chunks] if source_chunks else []

            # Skip empty
            if not head or not tail or not relation:
                continue

            key = f"{head}|{relation}|{tail}"
            if key not in seen:
                # First occurrence: create deduped entry
                deduped_triplet = {
                    "head": head.title(),
                    "relation": relation.replace("_", " ").title(),
                    "tail": tail.title(),
                    "source_chunks": list(set(source_chunks)),  # Remove duplicates
                }
                seen[key] = deduped_triplet
                deduped.append(deduped_triplet)
            else:
                # Merge source chunks from duplicate
                seen[key]["source_chunks"] = sorted(
                    list(set(seen[key]["source_chunks"] + source_chunks))
                )

        return deduped
