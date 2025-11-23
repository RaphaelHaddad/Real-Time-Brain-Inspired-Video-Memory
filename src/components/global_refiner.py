"""
Global Refinement: Merge & deduplicate local triplets, infer missing links
Uses lightweight LLM call to clean up and enrich the triplet graph
"""

import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..core.config import ChunkingConfig
from .prompts import get_llm_injector_prompt_template
from ..core.logger import get_logger

logger = get_logger(__name__)


class CompactTripletOutput(BaseModel):
    """Structured output for compact triplet format"""

    triplets: List[List] = Field(
        description="List of triplets as [head, relation, tail, source_chunks]"
    )


class GlobalRefiner:
    """Lightweight global refinement of merged triplets"""

    def __init__(self, llm_config, chunking_config: ChunkingConfig):
        self.chunking_config = chunking_config
        self.llm = ChatOpenAI(
            base_url=llm_config.endpoint,
            api_key=llm_config.api_key,
            model=llm_config.model_name,
            temperature=0.0,  # Deterministic refinement
            max_tokens=chunking_config.refinement_max_tokens,
        )

        self.chain = self.llm.with_structured_output(CompactTripletOutput)
        self.prompt_template = get_llm_injector_prompt_template()

        logger.info(
            f"Initialized GlobalRefiner with max_tokens={chunking_config.refinement_max_tokens}"
        )

    async def refine_triplets(
        self, triplets: List[Dict[str, Any]], network_info: str = "", 
        global_limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Refine merged triplets via lightweight LLM call

        Args:
            triplets: Local extracted triplets (may include source_chunks)
            network_info: Optional graph context
            global_limit: Max triplets to process (prevents huge prompts)

        Returns:
            Refined triplet list with source_chunks preserved/updated
        """
        import time
        refine_start = time.perf_counter()
        
        try:
            if not triplets:
                logger.debug("No triplets to refine")
                return []

            # Cap input to global limit to keep prompt small
            capped_triplets = triplets[:global_limit]
            if len(triplets) > global_limit:
                logger.info(
                    f"Capping refinement input: {len(triplets)} → {global_limit} triplets"
                )

            # Format triplets for prompt (compact format to save tokens)
            triplets_json = json.dumps(capped_triplets, indent=None)  # No indent = more compact

            prompt = self.prompt_template.format(
                network_info=network_info or "No graph context available",
                pre_extracted_triplets=triplets_json,
            )

            prompt_words = len(prompt.split())
            logger.debug(
                f"Refining {len(capped_triplets)} triplets "
                f"(prompt ~{prompt_words} words, ~{prompt_words * 1.3:.0f} tokens)"
            )

            # Call structured LLM
            logger.debug("Starting global refinement LLM call...")
            llm_start = time.perf_counter()
            output = await self.chain.ainvoke([("user", prompt)])
            llm_time = time.perf_counter() - llm_start
            logger.debug(f"LLM call completed in {llm_time:.2f}s")

            # Convert compact list format to dict format
            compact_triplets = output.triplets
            logger.debug(f"LLM refinement output: {json.dumps(compact_triplets, indent=2)}")
            
            refined = []
            for item in compact_triplets:
                if len(item) == 4:
                    triplet_dict = {
                        "head": item[0],
                        "relation": item[1],
                        "tail": item[2],
                        "source_chunks": item[3] if isinstance(item[3], list) else [item[3]]
                    }
                    refined.append(triplet_dict)
            
            # Post-process: ensure source_chunks field exists and is properly formatted
            for triplet in refined:
                if "source_chunks" not in triplet or not triplet["source_chunks"]:
                    # If LLM didn't provide source_chunks, try fuzzy match against input
                    triplet["source_chunks"] = self._backtrack_chunk_indices(
                        triplet, capped_triplets
                    )
                else:
                    # Ensure it's a sorted list of unique integers
                    if isinstance(triplet["source_chunks"], list):
                        triplet["source_chunks"] = sorted(list(set(triplet["source_chunks"])))
            
            refine_time = time.perf_counter() - refine_start
            logger.info(
                f"Refinement complete: {len(capped_triplets)} → {len(refined)} triplets (total: {refine_time:.2f}s, LLM: {llm_time:.2f}s)"
            )

            return refined

        except Exception as e:
            refine_time = time.perf_counter() - refine_start
            logger.error(f"Global refinement failed after {refine_time:.2f}s: {str(e)}")
            # Fallback: return deduplicated input (capped) with merged chunks
            deduped = self._deduplicate_triplets(triplets[:global_limit])
            logger.info(f"Using fallback deduplication: {len(triplets[:global_limit])} → {len(deduped)} triplets")
            return deduped

    def _backtrack_chunk_indices(
        self, refined_triplet: Dict[str, Any], original_triplets: List[Dict[str, Any]]
    ) -> List[int]:
        """Backtrack refined triplet to original chunks via fuzzy matching"""
        from difflib import SequenceMatcher
        
        refined_key = f"{refined_triplet.get('head','').lower()}|{refined_triplet.get('relation','').lower()}|{refined_triplet.get('tail','').lower()}"
        
        # Try exact match first
        for orig in original_triplets:
            orig_key = f"{orig.get('head','').lower()}|{orig.get('relation','').lower()}|{orig.get('tail','').lower()}"
            if orig_key == refined_key:
                return orig.get("source_chunks", [])
        
        # Fuzzy match: find closest original triplet
        best_match_chunks = []
        best_ratio = 0.0
        for orig in original_triplets:
            orig_key = f"{orig.get('head','').lower()}|{orig.get('relation','').lower()}|{orig.get('tail','').lower()}"
            ratio = SequenceMatcher(None, refined_key, orig_key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_chunks = orig.get("source_chunks", [])
        
        if best_ratio > 0.7:  # Fuzzy match threshold
            logger.debug(f"Backtracked refined triplet to chunks {best_match_chunks} (ratio: {best_ratio:.2f})")
            return best_match_chunks
        else:
            logger.warning(f"Could not backtrack refined triplet to source chunks: {refined_triplet}")
            return []

    def _deduplicate_triplets(
        self, triplets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple deduplication fallback, preserving source_chunks"""
        seen = {}  # Maps key → merged triplet
        deduped = []

        for triplet in triplets:
            head = triplet.get("head", "").strip().lower()
            relation = triplet.get("relation", "").strip().lower()
            tail = triplet.get("tail", "").strip().lower()
            source_chunks = triplet.get("source_chunks", [])
            
            if not isinstance(source_chunks, list):
                source_chunks = [source_chunks] if source_chunks else []

            if not head or not tail or not relation:
                continue

            key = f"{head}|{relation}|{tail}"
            if key not in seen:
                deduped_entry = {
                    "head": triplet.get("head", ""),
                    "relation": triplet.get("relation", ""),
                    "tail": triplet.get("tail", ""),
                    "source_chunks": list(set(source_chunks)),
                }
                seen[key] = deduped_entry
                deduped.append(deduped_entry)
            else:
                # Merge source chunks
                seen[key]["source_chunks"] = sorted(
                    list(set(seen[key]["source_chunks"] + source_chunks))
                )

        return deduped
