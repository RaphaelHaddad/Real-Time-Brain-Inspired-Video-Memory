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
from ..core.config import ChunkingConfig
from ..core.logger import get_logger

logger = get_logger(__name__)


class PreLLMInjector:
    """Local extraction from chunks using optimized pipe-delimited LLM calls"""

    def __init__(self, llm: BaseChatModel, config: ChunkingConfig):
        self.config = config
        self.llm = llm
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
            f"max_triplets_per_chunk={config.max_triplets_per_chunk}"
        )

    async def extract_local_triplets(
        self, content: str, network_info: str = ""
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

            # Log a sample of the first few chunks (truncated) to aid debugging
            for i, c in enumerate(chunks[:3]):
                short = (c[:400] + '...') if len(c) > 400 else c
                logger.debug(f"Chunk[{i}] ~{len(c.split())} words, sample: {short}")

            if not chunks:
                logger.warning("No chunks produced from content")
                return []

            # Extract triplets from each chunk in parallel (now returns triplets with chunk indices)
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
        """Extract triplets from multiple chunks in parallel, tracking chunk indices"""
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

    async def _extract_chunk_triplets(self, chunk: str, chunk_idx: int = 0) -> List[Dict[str, Any]]:
        """Extract triplets from a single chunk using optimized pipe-delimited LLM call"""
        try:
            # Debug: log the chunk size before calling LLM
            logger.debug(f"Calling LLM on chunk {chunk_idx} (~{len(chunk.split())} words)")

            # Format the prompt with the chunk content
            prompt = self.prompt_template.format(input=chunk)

            # Call the LLM directly
            response = await self.chain.ainvoke([("user", prompt)])
            llm_output = getattr(response, 'content', str(response)).strip()

            # Parse the pipe-delimited output
            triplets = self._parse_pipe_delimited_output(llm_output, chunk_idx)

            logger.debug(f"Extracted {len(triplets)} triplets from chunk {chunk_idx}")
            return triplets

        except Exception as e:
            logger.error(f"Chunk {chunk_idx} extraction error: {str(e)}")
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
