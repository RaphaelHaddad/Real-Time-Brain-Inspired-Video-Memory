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
from .prompts import get_llm_injector_prompt_template, get_llm_injector_instruction_prompt_template
from ..core.logger import get_logger

logger = get_logger(__name__)


class CompactTripletOutput(BaseModel):
    """Structured output for compact triplet format"""

    triplets: List[List] = Field(
        description="List of triplets as [head, relation, tail, source_chunks]"
    )


class InstructionRefinementOutput(BaseModel):
    """Structured output for instruction-based refinement"""

    new_triplets: List[List] = Field(
        description="New triplets not present in context"
    )
    inter_chunk_relations: List[List] = Field(
        description="Relations between entities from different chunks"
    )
    merge_instructions: List[Dict[str, str]] = Field(
        description="Instructions to merge local duplicates into existing entities"
    )
    prune_instructions: List[Dict[str, str]] = Field(
        description="Instructions to remove noise or refuted facts from context"
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

        # Create both structured and raw chains for fallback
        self.instruction_chain = self.llm.with_structured_output(InstructionRefinementOutput)
        self.instruction_chain_raw = self.llm  # Raw chain for direct text output
        self.instruction_prompt_template = get_llm_injector_instruction_prompt_template()

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
                # Ensure template receives limits to avoid KeyErrors if placeholders are present
                max_new_triplets=getattr(self.chunking_config, 'max_new_triplets', global_limit),
            )

            logger.debug(f"Global refinement full input prompt: {prompt}")

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

            logger.debug(f"Global refinement raw LLM output: {output}")

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

    def _get_short_chunk_id(self, full_chunk_id: str) -> str:
        """Extract short ID {batch}_{chunk} from full chunk_id"""
        parts = full_chunk_id.split('_')
        if len(parts) >= 3:
            return f"{parts[-2]}_{parts[-1]}"
        return "?"

    async def refine_triplets_instruction_based(
        self, triplets: List[Dict[str, Any]], subgraphs: Dict[str, str] = None,
        global_limit: int = 25
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Advanced instruction-based refinement with operations and subgraph context

        Args:
            triplets: Local extracted triplets
            subgraphs: Dict of subgraph strings keyed by chunk_id
            global_limit: Max triplets to process

        Returns:
            Refined triplet list with operations applied
        """
        import time
        refine_start = time.perf_counter()
        
        try:
            if not triplets:
                logger.debug("No triplets to refine")
                return []

            # Cap input to global limit
            capped_triplets = triplets[:global_limit]
            if len(triplets) > global_limit:
                logger.info(
                    f"Capping refinement input: {len(triplets)} → {global_limit} triplets"
                )

            # Format subgraphs context
            subgraph_context = ""
            context_is_empty = False
            if subgraphs:
                subgraph_list = [f"Subgraph_{i+1}: {subg}" for i, (cid, subg) in enumerate(subgraphs.items())]
                subgraph_context = "\n".join(subgraph_list[:5])  # Limit to top 5 subgraphs
            else:
                subgraph_context = "No subgraph context available"
                context_is_empty = True

            # Set the empty context rule conditionally
            if context_is_empty:
                empty_context_rule = "RULE: If CONTEXT SUBGRAPHS is empty or says \"No subgraph context available\", set \"inter_chunk_relations\", \"merge_instructions\", and \"prune_instructions\" to []. ONLY add to \"new_triplets\" from NEW CANDIDATES. DO NOT invent IDs, entities, or relations—stick to provided data. For merges/links, match EXACT entity names/IDs from CONTEXT only."
            else:
                empty_context_rule = ""

            # Pre-LLM validation: inject flag for empty context to prevent hallucinations
            if context_is_empty:
                subgraph_context += "\nNo existing entities or relations."

            # Format triplets for prompt (convert source_chunks to short format)
            processed_triplets = []
            for triplet in capped_triplets:
                processed_triplet = triplet.copy()
                if "source_chunks" in processed_triplet and processed_triplet["source_chunks"]:
                    # Convert full UUID format to short batch_chunk format
                    short_chunks = []
                    for chunk_id in processed_triplet["source_chunks"]:
                        if isinstance(chunk_id, str):
                            short_id = self._get_short_chunk_id(chunk_id)
                            short_chunks.append(short_id)
                        else:
                            short_chunks.append(chunk_id)
                    processed_triplet["source_chunks"] = short_chunks
                processed_triplets.append(processed_triplet)
            
            triplets_json = json.dumps(processed_triplets, indent=None)

            prompt = self.instruction_prompt_template.format(
                empty_context_rule=empty_context_rule,
                subgraph_context=subgraph_context,
                pre_extracted_triplets=triplets_json,
                max_new_triplets=getattr(self.chunking_config, 'max_new_triplets', 15),
                max_inter_chunk_relations=getattr(self.chunking_config, 'max_inter_chunk_relations', 15),
                max_merge_instructions=getattr(self.chunking_config, 'max_merge_instructions', 10),
                max_prune_instructions=getattr(self.chunking_config, 'max_prune_instructions', 10),
            )

            logger.debug(f"Instruction-based refinement full input prompt: {prompt}")

            prompt_words = len(prompt.split())
            logger.debug(
                f"Instruction-based refining {len(capped_triplets)} triplets "
                f"(prompt ~{prompt_words} words, ~{prompt_words * 1.3:.0f} tokens)"
            )

            # Call raw LLM directly (more reliable than structured output)
            logger.debug("Starting raw LLM call for instruction-based refinement...")
            llm_start = time.perf_counter()
            
            from langchain_core.messages import HumanMessage
            raw_output = await self.instruction_chain_raw.ainvoke([HumanMessage(content=prompt)])
            
            llm_time = time.perf_counter() - llm_start
            logger.debug(f"Raw LLM call completed in {llm_time:.2f}s")
            
            # Extract text content from response
            if hasattr(raw_output, 'content'):
                raw_text = raw_output.content
            else:
                raw_text = str(raw_output)
            
            logger.debug(f"Raw LLM response ({len(raw_text)} chars): {raw_text[:500]}...")

            # Parse JSON from raw text
            def _parse_json_from_text(text: str) -> Dict[str, Any]:
                """Parse JSON from raw LLM text response."""
                import re
                parsed = {
                    "new_triplets": [],
                    "inter_chunk_relations": [],
                    "merge_instructions": [],
                    "prune_instructions": [],
                }
                try:
                    # Try to find JSON object in the text
                    # First try direct parse
                    try:
                        raw_json = json.loads(text.strip())
                    except json.JSONDecodeError:
                        # Try to extract JSON from markdown code blocks or surrounding text
                        json_match = re.search(r'\{[\s\S]*\}', text)
                        if json_match:
                            raw_json = json.loads(json_match.group(0))
                        else:
                            logger.warning(f"No JSON found in LLM response")
                            return parsed
                    
                    parsed["new_triplets"] = raw_json.get("new_triplets", [])
                    parsed["inter_chunk_relations"] = raw_json.get("inter_chunk_relations", [])
                    parsed["merge_instructions"] = raw_json.get("merge_instructions", [])
                    parsed["prune_instructions"] = raw_json.get("prune_instructions", [])
                    logger.debug(f"Successfully parsed JSON: {len(parsed['new_triplets'])} new_triplets, {len(parsed['inter_chunk_relations'])} inter_chunk, {len(parsed['merge_instructions'])} merges, {len(parsed['prune_instructions'])} prunes")
                    return parsed
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}")
                    logger.debug(f"Failed to parse text: {text[:500]}")
                    return parsed
            
            parsed_op = _parse_json_from_text(raw_text)

            # Post-LLM sanitization: if context was empty but LLM hallucinated operations, force empty
            if context_is_empty:
                if parsed_op.get("inter_chunk_relations"):
                    logger.warning(f"LLM hallucinated {len(parsed_op['inter_chunk_relations'])} inter_chunk_relations despite empty context, forcing to []")
                    parsed_op["inter_chunk_relations"] = []
                if parsed_op.get("merge_instructions"):
                    logger.warning(f"LLM hallucinated {len(parsed_op['merge_instructions'])} merge_instructions despite empty context, forcing to []")
                    parsed_op["merge_instructions"] = []
                if parsed_op.get("prune_instructions"):
                    logger.warning(f"LLM hallucinated {len(parsed_op['prune_instructions'])} prune_instructions despite empty context, forcing to []")
                    parsed_op["prune_instructions"] = []

            # Log concise summaries for inspection (AFTER sanitization)
            def _log_list(name, items):
                count = len(items) if items else 0
                sample = items[:3] if items else []
                logger.debug(f"Instruction result '{name}': count={count}, sample={sample}")

            _log_list("new_triplets", parsed_op.get("new_triplets", []))
            _log_list("inter_chunk_relations", parsed_op.get("inter_chunk_relations", []))
            _log_list("merge_instructions", parsed_op.get("merge_instructions", []))
            _log_list("prune_instructions", parsed_op.get("prune_instructions", []))

            # Process operations (using parsed_op for speed where possible)
            refined = []
            
            # Add new_triplets
            for item in parsed_op.get("new_triplets", []) or []:
                if len(item) >= 3:
                    triplet_dict = {
                        "head": item[0],
                        "relation": item[1],
                        "tail": item[2],
                        "source_chunks": item[3] if len(item) > 3 and isinstance(item[3], list) else []
                    }
                    refined.append(triplet_dict)

            # Add inter_chunk_relations
            for item in parsed_op.get("inter_chunk_relations", []) or []:
                if len(item) >= 3:
                    triplet_dict = {
                        "head": item[0],
                        "relation": item[1],
                        "tail": item[2],
                        "source_chunks": item[3] if len(item) > 3 and isinstance(item[3], list) else []
                    }
                    refined.append(triplet_dict)

            # Log merge_instructions (not yet implemented)
            if parsed_op.get("merge_instructions"):
                logger.info(f"LLM suggested {len(parsed_op.get('merge_instructions'))} entity merges: {parsed_op.get('merge_instructions')}")
                # TODO: Implement actual node merging logic

            # Log prune_instructions (not yet implemented)  
            if parsed_op.get("prune_instructions"):
                logger.info(f"LLM suggested {len(parsed_op.get('prune_instructions'))} triplets to prune: {parsed_op.get('prune_instructions')}")
                # TODO: Implement actual pruning logic

            refine_time = time.perf_counter() - refine_start
            logger.info(
                f"Instruction-based refinement complete: {len(capped_triplets)} → {len(refined)} triplets (total: {refine_time:.2f}s, LLM: {llm_time:.2f}s)"
            )

            # Return both refined triplets and the parsed operations so caller can apply DB changes
            return refined, parsed_op

        except Exception as e:
            refine_time = time.perf_counter() - refine_start
            logger.error(f"Instruction-based refinement failed after {refine_time:.2f}s: {str(e)}")
            # Fallback: return deduplicated input
            deduped = self._deduplicate_triplets(triplets[:global_limit])
            logger.info(f"Using fallback deduplication: {len(triplets[:global_limit])} → {len(deduped)} triplets")
            return deduped, {
                "new_triplets": [],
                "inter_chunk_relations": [],
                "merge_instructions": [],
                "prune_instructions": [],
            }
