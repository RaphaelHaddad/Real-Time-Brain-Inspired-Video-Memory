"""
Pre-LLM Injector: Hierarchical chunking & local triplet extraction
Uses TokenTextSplitter + LLMGraphTransformer for efficient local extraction
Supports community-based question generation when enabled.
"""

import asyncio
import json
import random
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from .prompts import build_pre_llm_prompt_template, build_pre_llm_with_questions_prompt_template
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from ..core.config import ChunkingConfig, EmbedderConfig, LLMInjectorConfig, CommunityHighGraphConfig
from ..core.logger import get_logger

logger = get_logger(__name__)


class PreLLMInjector:
    """Local extraction from chunks using optimized pipe-delimited LLM calls"""

    def __init__(self, llm: BaseChatModel, config: ChunkingConfig, embedder_config: EmbedderConfig = None, 
                 llm_injector_config: LLMInjectorConfig = None, community_config: CommunityHighGraphConfig = None):
        self.config = config
        self.llm = llm
        self.embedder_config = embedder_config
        self.llm_injector_config = llm_injector_config
        self.community_config = community_config
        # Holds details from the last extraction for external tracing/logging
        self.last_chunk_details: List[Dict[str, Any]] = []
        
        # Check if community-based question generation is enabled
        self._community_enabled = community_config and community_config.community_creator
        self._questions_per_chunk = community_config.question_per_chunk if community_config else 2
        
        # Build prompt templates based on whether community features are enabled
        if self._community_enabled:
            # Use combined prompt for efficiency (single LLM call for triplets + questions)
            self.prompt_template = build_pre_llm_with_questions_prompt_template(
                config.max_triplets_per_chunk, self._questions_per_chunk
            )
            logger.info(f"Community features ENABLED: generating {self._questions_per_chunk} questions per chunk")
        else:
            # Use original triplet-only prompt
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
            f"subgraph_extraction={llm_injector_config.subgraph_extraction_injection if llm_injector_config else False}, "
            f"community_creator={self._community_enabled}"
        )

    def _truncate_text(self, text: str, max_words: int = 25) -> str:
        """Return a truncated version of the text using at most max_words words."""
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    async def extract_local_triplets(
        self, content: str, network_info: str = "", neo4j_handler = None, batch_idx: int = 0, run_uuid: str = ""
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str], Dict[str, List[str]]]:
        """
        Extract triplets from content using hierarchical chunking + local extraction.
        When community_creator is enabled, also extracts questions from each chunk.

        Args:
            content: Aggregated VLM output text
            network_info: Optional graph context (unused in local extraction)
            neo4j_handler: Handler for graph operations (similarity search)
            batch_idx: Current batch index
            run_uuid: Unique identifier for the run

        Returns:
            Tuple of:
            - List of triplet dicts with 'head', 'relation', 'tail', 'source_chunks'
            - List of chunk dicts with 'id', 'content', 'embedding' (if available)
            - Dict of subgraph strings keyed by chunk_id (for similar chunks)
            - Dict of questions keyed by chunk_id (only when community_creator is enabled)
        """
        try:
            # Split content into manageable chunks
            chunks_text = self.splitter.split_text(content)
            logger.debug(
                f"Split content into {len(chunks_text)} chunks (chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
            )

            # Generate IDs for chunks
            chunk_data = []
            for i, text in enumerate(chunks_text):
                chunk_id = f"{run_uuid}_{batch_idx}_{i}" if run_uuid else f"chunk_{batch_idx}_{i}"
                chunk_data.append({
                    "id": chunk_id,
                    "content": text,
                    "index": i,
                    "embedding": None
                })
                truncated = self._truncate_text(text, max_words=25)
                logger.debug(f"Chunk[{i}] ID={chunk_id} ~{len(text.split())} words, content: {truncated}")

            if not chunks_text:
                logger.warning("No chunks produced from content")
                return [], [], {}, {}

            # Extract triplets (and optionally questions) from each chunk
            chunk_questions: Dict[str, List[str]] = {}
            
            if self.llm_injector_config and self.llm_injector_config.subgraph_extraction_injection:
                triplets, updated_chunk_data, subgraphs = await self._parallel_chunk_extraction_with_similarity(chunk_data, neo4j_handler)
                # Extract questions from last_chunk_details if community is enabled
                if self._community_enabled:
                    for detail in self.last_chunk_details:
                        questions = detail.get('questions', [])
                        if questions:
                            chunk_questions[detail['chunk_id']] = questions
            else:
                triplets = await self._parallel_chunk_extraction(chunk_data)
                updated_chunk_data = chunk_data
                subgraphs = {}
                # Extract questions from last_chunk_details if community is enabled
                if self._community_enabled:
                    for detail in self.last_chunk_details:
                        questions = detail.get('questions', [])
                        if questions:
                            chunk_questions[detail['chunk_id']] = questions

            # Deduplicate at local level (preserves source_chunks)
            triplets = self._deduplicate_triplets(triplets)

            logger.info(
                f"Extracted {len(triplets)} local triplets from {len(chunks_text)} chunks"
            )
            if self._community_enabled:
                total_questions = sum(len(q) for q in chunk_questions.values())
                logger.info(f"Generated {total_questions} questions across {len(chunk_questions)} chunks")
                
            return triplets, updated_chunk_data, subgraphs, chunk_questions

        except Exception as e:
            logger.error(f"Pre-LLM extraction failed: {str(e)}")
            return [], [], {}, {}

    async def _parallel_chunk_extraction(
        self, chunk_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract triplets from multiple chunks in parallel or serially based on config"""
        chunks = [c["content"] for c in chunk_data]
        chunk_ids = [c["id"] for c in chunk_data]
        
        if not self.config.batch_llm_parallelism:
            # Serial execution
            logger.debug("Using serial LLM extraction (batch_llm_parallelism=false)")
            all_triplets: List[Dict[str, Any]] = []
            self.last_chunk_details = []
            for i, chunk in enumerate(chunks):
                try:
                    triplets, questions = await self._extract_chunk_triplets(chunk, i, chunk_ids[i])
                    logger.debug(f"Chunk {i} returned {len(triplets)} triplets, {len(questions)} questions")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_id": chunk_ids[i],
                        "chunk_text": chunk,
                        "triplets": triplets,
                        "questions": questions,
                    })
                    all_triplets.extend(triplets)
                except Exception as e:
                    logger.warning(f"Chunk {i} extraction failed: {e}")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_id": chunk_ids[i],
                        "chunk_text": chunk,
                        "triplets": [],
                        "questions": [],
                        "error": str(e),
                    })
            return all_triplets
        
        # Parallel execution (original logic)
        # Limit concurrency to avoid overwhelming the LLM
        semaphore = asyncio.Semaphore(self.config.parallel_count)

        async def extract_with_semaphore(chunk_idx: int, chunk: str, chunk_id: str) -> tuple[List[Dict[str, Any]], List[str]]:
            async with semaphore:
                return await self._extract_chunk_triplets(chunk, chunk_idx, chunk_id)

        # Launch parallel tasks with chunk indices
        tasks = [extract_with_semaphore(i, chunk, chunk_ids[i]) for i, chunk in enumerate(chunks)]
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
                    "chunk_id": chunk_ids[i],
                    "chunk_text": chunks[i],
                    "triplets": [],
                    "questions": [],
                    "error": str(result),
                })
            else:
                triplets, questions = result
                logger.debug(f"Chunk {i} returned {len(triplets)} triplets, {len(questions)} questions")
                self.last_chunk_details.append({
                    "chunk_index": i,
                    "chunk_id": chunk_ids[i],
                    "chunk_text": chunks[i],
                    "triplets": triplets,
                    "questions": questions,
                })
                all_triplets.extend(triplets)

        return all_triplets

    async def _parallel_chunk_extraction_with_similarity(self, chunk_data: List[Dict[str, Any]], neo4j_handler) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
        """Extract triplets with subgraph similarity calculation"""
        import asyncio
        from langchain_openai import OpenAIEmbeddings
        
        chunks = [c["content"] for c in chunk_data]
        chunk_ids = [c["id"] for c in chunk_data]

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
        
        # Update chunk_data with embeddings
        for i, emb in enumerate(chunk_embeddings):
            if not isinstance(emb, Exception):
                chunk_data[i]["embedding"] = emb
            else:
                logger.warning(f"Embedding failed for chunk {i}: {emb}")

        # Calculate similarities against existing batch nodes
        subgraphs = {}
        if neo4j_handler:
            batch_similarities = await self._calculate_batch_similarities(chunk_embeddings, neo4j_handler)
            
            # Concatenate per-chunk similarity lists, deduplicate by chunk id taking the MAX score,
            # then sort by score and take the top_k_similar_batch elements.
            # This produces a final, unique list of (chunk_id, score) as requested.
            final_scores = {}
            for chunk_similarities in batch_similarities:
                for chunk_id, score in chunk_similarities:
                    # keep the highest score observed for this chunk_id across all new chunks
                    if chunk_id not in final_scores or score > final_scores[chunk_id]:
                        final_scores[chunk_id] = score

            # Convert to list and sort by descending score
            final_score_list = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep only the top-k configured similar batches/chunks
            top_similar_chunks = final_score_list[: self.embedder_config.top_k_similar_batch]

            # Log the final deduplicated sorted list
            logger.debug(f"FINAL SIMILARITY LIST : {[(cid, f'{score:.3f}') for cid, score in top_similar_chunks]}")

            # Spawn parallel subgraph extraction tasks for each top similar chunk id
            try:
                sem = asyncio.Semaphore(self.config.parallel_count)

                async def extract_and_log(chunk_id: str):
                    async with sem:
                        subg = await self._extract_subgraph_for_chunk_id(chunk_id, neo4j_handler)
                        if subg:
                            logger.debug(f"{subg}")
                            subgraphs[chunk_id] = subg

                extract_tasks = [extract_and_log(cid) for cid, _ in top_similar_chunks]
                # Fire-and-forget - but gather to ensure logs appear during this run
                await asyncio.gather(*extract_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Failed to extract subgraphs for similar chunks: {e}")
            
            # Log per-chunk similarities
            for chunk_idx, similarities in enumerate(batch_similarities):
                if similarities:
                    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:self.embedder_config.top_k_similar_batch]
                    logger.debug(f"Chunk {chunk_idx} top similar chunks: {[(cid, f'{score:.3f}') for cid, score in top_similar]}")
                else:
                    logger.debug(f"Chunk {chunk_idx} top similar chunks: none (no existing chunks to compare)")
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
                    triplets, questions = await self._extract_chunk_triplets(chunk, i, chunk_ids[i])
                    logger.debug(f"Chunk {i} returned {len(triplets)} triplets, {len(questions)} questions")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_id": chunk_ids[i],
                        "chunk_text": chunk,
                        "triplets": triplets,
                        "questions": questions,
                    })
                    all_triplets.extend(triplets)
                except Exception as e:
                    logger.warning(f"Chunk {i} extraction failed: {e}")
                    self.last_chunk_details.append({
                        "chunk_index": i,
                        "chunk_id": chunk_ids[i],
                        "chunk_text": chunk,
                        "triplets": [],
                        "questions": [],
                        "error": str(e),
                    })
            return all_triplets, chunk_data, subgraphs
        
        # Parallel LLM extraction (original logic)
        semaphore = asyncio.Semaphore(self.config.parallel_count)
        async def extract_with_semaphore(chunk_idx: int, chunk: str, chunk_id: str) -> tuple[List[Dict[str, Any]], List[str]]:
            async with semaphore:
                return await self._extract_chunk_triplets(chunk, chunk_idx, chunk_id)
        
        llm_tasks = [extract_with_semaphore(i, chunk, chunk_ids[i]) for i, chunk in enumerate(chunks)]
        
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
                    "chunk_id": chunk_ids[i],
                    "chunk_text": chunks[i],
                    "triplets": [],
                    "questions": [],
                    "error": str(result),
                })
            else:
                triplets, questions = result
                logger.debug(f"Chunk {i} returned {len(triplets)} triplets, {len(questions)} questions")
                self.last_chunk_details.append({
                    "chunk_index": i,
                    "chunk_id": chunk_ids[i],
                    "chunk_text": chunks[i],
                    "triplets": triplets,
                    "questions": questions,
                })
                all_triplets.extend(triplets)

        return all_triplets, chunk_data, subgraphs

    async def _calculate_batch_similarities(self, chunk_embeddings, neo4j_handler) -> List[List[tuple[str, float]]]:
        """Calculate similarity between new chunks and existing Chunk nodes (per VLM inference)
        Returns a list (one per new chunk) of (existing_chunk_id, score) tuples.
        """
        import math
        
        # Query existing chunk nodes and their embeddings (keyed by chunk id, not batch id)
        existing_chunk_embeddings = await self._get_chunk_embeddings(neo4j_handler)
        
        similarities = []
        for chunk_emb in chunk_embeddings:
            if isinstance(chunk_emb, Exception):
                similarities.append([])
                continue
            
            chunk_similarities = []
            for chunk_id, existing_emb in existing_chunk_embeddings.items():
                if existing_emb:
                    # Calculate cosine similarity manually
                    similarity = self._cosine_similarity(chunk_emb, existing_emb)
                    chunk_similarities.append((chunk_id, float(similarity)))
            
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

    async def _get_chunk_embeddings(self, neo4j_handler) -> Dict[str, List[float]]:
        """Get embeddings for existing chunk nodes (keyed by chunk node id) in current graph"""
        try:
            async with neo4j_handler.driver.session() as session:
                result = await session.run("""
                    MATCH (c:Chunk:GraphNode)
                    WHERE c.graph_uuid = $graph_uuid AND c.id IS NOT NULL AND c.embedding IS NOT NULL
                    RETURN c.id as chunk_id, c.embedding as embedding
                    LIMIT 5000
                """, graph_uuid=neo4j_handler.run_uuid)

                chunk_embeddings = {}
                async for record in result:
                    chunk_id = record["chunk_id"]
                    embedding = record["embedding"]
                    if isinstance(embedding, list) and chunk_id:
                        chunk_embeddings[chunk_id] = embedding

                logger.debug(f"Found {len(chunk_embeddings)} existing chunk embeddings for similarity lookup")
                return chunk_embeddings
        except Exception as e:
            logger.warning(f"Failed to get chunk embeddings: {e}")
            return {}

    def _get_short_chunk_id(self, full_chunk_id: str) -> str:
        """Extract short ID {batch}_{chunk} from full chunk_id"""
        parts = full_chunk_id.split('_')
        if len(parts) >= 3:
            return f"{parts[-2]}_{parts[-1]}"
        return "?"

    async def _extract_subgraph_for_chunk_id(self, chunk_id: str, neo4j_handler) -> str:
        """Extract a concise, LLM-friendly subgraph surrounding the given chunk_id.

        The subgraph includes:
        - All Entity nodes directly connected to the given Chunk (via FROM_CHUNK)
        - All relationships where at least one endpoint is in that set of Entity nodes
        - All entities are shown with their source chunk IDs in format (EntityName / ID: batch_chunk)

        Returns a compact string representation, or an empty string if nothing found.
        """
        try:
            async with neo4j_handler.driver.session() as session:
                # Get entities connected to this chunk via source_chunk_ids metadata
                res = await session.run(
                    """
                    MATCH (e:Entity:GraphNode)
                    WHERE e.graph_uuid = $graph_uuid
                      AND $chunk_id IN coalesce(e.source_chunk_ids, [])
                    RETURN collect(DISTINCT e.name) AS entities
                    """,
                    chunk_id=chunk_id, graph_uuid=neo4j_handler.run_uuid,
                )
                record = await res.single()
                if not record:
                    logger.debug(f"No entities record found for chunk_id={chunk_id}")
                    return ""
                entity_names = record["entities"] if record["entities"] is not None else []
                if not entity_names:
                    logger.debug(f"No entities linked to chunk_id={chunk_id} via source_chunk_ids")
                    return ""
                logger.debug(f"Found {len(entity_names)} entities linked to chunk_id={chunk_id}")

                # Get max_connection_subgraph parameter from config
                max_connections = getattr(self.config, 'max_connection_subgraph', 2)
                logger.debug(f"Using max_connection_subgraph limit: {max_connections}")

                # Fetch relationships where at least one endpoint is in our entity set
                rel_query = """
                MATCH (e1:Entity:GraphNode)-[r]->(e2:Entity:GraphNode)
                WHERE e1.graph_uuid = $graph_uuid AND e2.graph_uuid = $graph_uuid
                AND (e1.name IN $entity_names OR e2.name IN $entity_names)
                RETURN e1.name as head, type(r) as rel, e2.name as tail,
                       coalesce(e1.source_chunk_ids, []) as head_chunks,
                       coalesce(e2.source_chunk_ids, []) as tail_chunks
                """
                res = await session.run(
                    rel_query, graph_uuid=neo4j_handler.run_uuid, entity_names=entity_names
                )

                all_relations = []
                async for rec in res:
                    head = rec["head"]
                    tail = rec["tail"]
                    rel = rec["rel"]
                    head_chunks = rec["head_chunks"] if rec["head_chunks"] is not None else []
                    tail_chunks = rec["tail_chunks"] if rec["tail_chunks"] is not None else []

                    all_relations.append((head, rel, tail, head_chunks, tail_chunks))

                # Apply connection limit per seed entity
                relations = self._limit_subgraph_connections(all_relations, entity_names, max_connections)

                if not relations:
                    logger.debug(f"No relationships found for entities linked to chunk_id={chunk_id} after connection limiting")

                # Build a compact string representation
                # Entities list removed as per request

                # Relationships: head-rel->tail
                rel_parts = []
                for head, rel, tail, head_chunks, tail_chunks in relations:
                    # Get short ID for head entity
                    head_short_id = self._get_short_chunk_id(head_chunks[0]) if head_chunks else "?"
                    head_formatted = f"({head} / ID: {head_short_id})"
                    
                    # Get short ID for tail entity  
                    tail_short_id = self._get_short_chunk_id(tail_chunks[0]) if tail_chunks else "?"
                    tail_formatted = f"({tail} / ID: {tail_short_id})"
                    
                    rel_parts.append(f"{head_formatted}-[{rel}]->{tail_formatted}")

                # Build final short string: Subgraph only
                rels_str = ", ".join(rel_parts)
                short = f"Subgraph: {rels_str}"
                return short
        except Exception as e:
            logger.warning(f"Failed to extract subgraph for chunk {chunk_id}: {e}")
            return ""

    async def _extract_chunk_triplets(self, chunk_text: str, chunk_idx: int, chunk_id: str) -> tuple[List[Dict], List[str]]:
        """
        Extracts triplets (and optionally questions) from a single chunk using the LLM with a STRICT TIMEOUT.
        
        Returns:
            Tuple of (triplets list, questions list). Questions list is empty if community features disabled.
        """
        prompt = self.prompt_template.invoke({"input": chunk_text})
        
        # TIMEOUT CONFIG: use values from chunking config (defaults keep previous behavior)
        timeout_seconds = getattr(self.config, 'chunk_timeout_seconds', 45.0)
        timeout_retries = int(getattr(self.config, 'chunk_timeout_retries', 0) or 0)

        logger.debug(f"Calling LLM on chunk {chunk_idx} (ID={chunk_id}) (~{len(chunk_text.split())} words) with timeout={timeout_seconds}s retries={timeout_retries}")

        # Try with retries on timeouts
        for attempt in range(timeout_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.chain.ainvoke(prompt),
                    timeout=timeout_seconds
                )

                # Parse the output
                llm_output = getattr(response, 'content', str(response)).strip()
                
                if self._community_enabled:
                    # Parse combined triplets + questions output
                    triplets, questions = self._parse_combined_output(llm_output, chunk_idx, chunk_id)
                    logger.debug(f"Extracted {len(triplets)} triplets and {len(questions)} questions from chunk {chunk_idx} (attempt {attempt+1})")
                    # Log questions at debug level
                    for q in questions:
                        logger.debug(f"  Question from chunk {chunk_idx}: {q}")
                    return triplets, questions
                else:
                    # Parse triplets only (original behavior)
                    triplets = self._parse_pipe_delimited_output(llm_output, chunk_idx, chunk_id)
                    logger.debug(f"Extracted {len(triplets)} triplets from chunk {chunk_idx} (attempt {attempt+1})")
                    return triplets, []

            except asyncio.TimeoutError:
                # If we have retries left, log and continue; otherwise skip
                if attempt < timeout_retries:
                    logger.warning(f"Chunk {chunk_idx} timed out after {timeout_seconds}s (attempt {attempt+1}/{timeout_retries+1}), retrying...")
                    # small backoff
                    try:
                        await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
                    except Exception:
                        pass
                    continue
                else:
                    logger.warning(f"⚠️ Chunk {chunk_idx} TIMED OUT after {timeout_seconds}s on final attempt. Skipping.")
                    return [], []

            except Exception as e:
                logger.error(f"Error extracting from chunk {chunk_idx} (attempt {attempt+1}): {e}")
                return [], []
        
        return [], []

    def _parse_combined_output(self, llm_output: str, chunk_idx: int, chunk_id: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Parse combined triplets + questions output from the LLM"""
        triplets = []
        questions = []
        
        # Split by sections
        lines = llm_output.strip().split('\n')
        
        # Track which section we're in
        in_triplets = False
        in_questions = False
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Detect section headers
            if 'triplets:' in line_lower or line_lower == 'triplets':
                in_triplets = True
                in_questions = False
                continue
            elif 'questions:' in line_lower or line_lower == 'questions':
                in_triplets = False
                in_questions = True
                continue
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Parse triplets (pipe-delimited)
            if in_triplets and '|' in line_stripped:
                parts = line_stripped.split('|')
                if len(parts) == 3:
                    head = parts[0].strip()
                    relation = parts[1].strip()
                    tail = parts[2].strip()
                    
                    if head and relation and tail:
                        triplet = {
                            "head": head,
                            "relation": relation,
                            "tail": tail,
                            "source_chunks": [chunk_id]
                        }
                        triplets.append(triplet)
                        
                        # Respect max_triplets_per_chunk limit
                        if len(triplets) >= self.config.max_triplets_per_chunk:
                            in_triplets = False
            
            # Parse questions (one per line)
            elif in_questions:
                # Clean up the question (remove numbering, bullets, etc.)
                question = line_stripped.lstrip('0123456789.-) ').strip()
                if question and len(question) > 5:  # Minimum question length
                    questions.append(question)
                    
                    # Respect questions_per_chunk limit
                    if len(questions) >= self._questions_per_chunk:
                        in_questions = False
        
        return triplets, questions

    def _parse_pipe_delimited_output(self, llm_output: str, chunk_idx: int, chunk_id: str) -> List[Dict[str, Any]]:
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
                        "source_chunks": [chunk_id]
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

    def _limit_subgraph_connections(self, all_relations: List[tuple], seed_entities: List[str], max_connections: int) -> List[tuple]:
        """Limit the number of outside connections per seed entity to prevent subgraph explosion.

        Args:
            all_relations: List of (head, rel, tail, head_chunks, tail_chunks) tuples
            seed_entities: List of entity names that are seeds (directly linked to chunk)
            max_connections: Maximum number of outside connections per seed entity

        Returns:
            Filtered list of relationships with connection limits applied
        """
        import random

        # Separate relationships into internal (between seed entities) and external (to outside entities)
        internal_relations = []
        external_relations_by_seed = {}  # seed_entity -> list of external relations

        seed_set = set(seed_entities)

        for relation in all_relations:
            head, rel, tail, head_chunks, tail_chunks = relation

            # Check if this is an internal relationship (both entities are seeds)
            if head in seed_set and tail in seed_set:
                internal_relations.append(relation)
            else:
                # External relationship - find which seed entity is involved
                if head in seed_set:
                    seed_entity = head
                elif tail in seed_set:
                    seed_entity = tail
                else:
                    # This shouldn't happen given our query, but skip if neither is a seed
                    continue

                if seed_entity not in external_relations_by_seed:
                    external_relations_by_seed[seed_entity] = []
                external_relations_by_seed[seed_entity].append(relation)

        # Apply connection limit per seed entity
        selected_external_relations = []
        for seed_entity, relations in external_relations_by_seed.items():
            if len(relations) <= max_connections:
                # Take all if under limit
                selected_external_relations.extend(relations)
            else:
                # Randomly select max_connections
                selected = random.sample(relations, max_connections)
                selected_external_relations.extend(selected)
                logger.debug(f"Limited {seed_entity} from {len(relations)} to {max_connections} external connections")

        # Combine internal and selected external relations
        final_relations = internal_relations + selected_external_relations

        logger.debug(f"Subgraph connection limiting: {len(all_relations)} total -> {len(final_relations)} "
                    f"(internal: {len(internal_relations)}, external: {len(selected_external_relations)})")

        return final_relations
