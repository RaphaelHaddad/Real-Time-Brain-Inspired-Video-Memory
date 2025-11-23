from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
from ..core.config import LLMInjectorConfig, ChunkingConfig
from .prompts import get_llm_injector_prompt_template
from ..core.logger import get_logger

logger = get_logger(__name__)


class Triplet(BaseModel):
    head: str = Field(description="The source entity")
    relation: str = Field(description="The relationship between entities")
    tail: str = Field(description="The target entity")


class GraphOutput(BaseModel):
    triplets: List[Triplet] = Field(description="List of extracted triplets")

class LLMInjector:
    def __init__(self, config: LLMInjectorConfig, chunking_config: Optional[ChunkingConfig] = None):
        self.config = config
        self.chunking_config = chunking_config
        
        # Use the generic ChatOpenAI class which works with any OpenAI-compatible API
        # IMPORTANT: Setting max_tokens to the full model context (e.g. 8192) triggers overflow
        # validation before we dynamically bind a safer completion value. Use a conservative
        # default here and override per-call.
        self.llm = ChatOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=min(512, config.max_tokens),  # conservative base
            verbose=False
        )

        # Use structured output for reliable parsing
        self.chain = self.llm.with_structured_output(GraphOutput)

        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        logger.info(f"Initialized LLMInjector (chunking: {chunking_config is not None and chunking_config.enabled})")

    def _load_prompt_template(self) -> ChatPromptTemplate:
        """Load and configure the prompt template"""
        # Use centralized injector prompt template for easier maintenance and
        # to ensure placeholders remain consistent.
        return get_llm_injector_prompt_template()

    async def extract_triplets(self, aggregated_content: str, network_info: str, 
                              pre_extracted_triplets: Optional[List[Dict[str, str]]] = None,
                              global_limit: int = 25,
                              trace_file_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract and refine triplets with optional pre-extraction context
        
        Args:
            aggregated_content: Full VLM output (used for context only if pre-extraction available)
            network_info: Graph statistics for context
            pre_extracted_triplets: Triplets from hierarchical chunking (if available)
            global_limit: Max triplets to process (prevents token limit errors)
        
        Returns:
            Final refined triplet list
        """
        try:
            # If pre-extraction available, use it to seed the refinement
            if pre_extracted_triplets is not None:
                # Cap to global limit
                capped_triplets = pre_extracted_triplets[:global_limit]
                if len(pre_extracted_triplets) > global_limit:
                    logger.info(
                        f"Capping final LLM input: {len(pre_extracted_triplets)} → {global_limit} triplets"
                    )
                
                logger.debug(f"Refining {len(capped_triplets)} pre-extracted triplets")
                # Provide max_new_triplets constraint to the prompt template when available
                max_new = getattr(self.chunking_config, 'max_new_triplets', 20)
                prompt = self.prompt_template.format(
                    network_info=network_info,
                    pre_extracted_triplets=json.dumps(capped_triplets, indent=None),  # Compact
                    max_new_triplets=max_new,
                )
            else:
                # Explicitly handle None (should not occur when chunking enabled)
                logger.warning("No pre-extracted triplets provided; proceeding with empty candidate set.")
                max_new = getattr(self.chunking_config, 'max_new_triplets', 20)
                prompt = self.prompt_template.format(
                    network_info=network_info,
                    pre_extracted_triplets="[]",
                    max_new_triplets=max_new,
                )

            prompt_words = len(prompt.split())
            logger.debug(f"LLM Prompt: ~{prompt_words} words (~{prompt_words * 1.3:.0f} tokens)")
            
            # Optionally write prompt to trace file
            if trace_file_path:
                try:
                    with open(trace_file_path, 'a', encoding='utf-8') as tf:
                        tf.write("\n===== FINAL LLM INJECTOR PROMPT =====\n")
                        tf.write(prompt)
                        tf.write("\n===== END PROMPT =====\n")
                except Exception as _:
                    pass

            # Calculate safe max_tokens to avoid context overflow
            estimated_prompt_tokens = int(prompt_words * 1.3)
            # Model context length assumed 8192; keep a wider margin since output is small
            safe_max_tokens = min(
                512,  # we rarely need more than a few hundred tokens for triplets
                self.config.max_tokens,
                8192 - estimated_prompt_tokens - 400  # larger safety margin
            )
            if safe_max_tokens < 100:
                raise ValueError(f"Prompt too long ({estimated_prompt_tokens} tokens), cannot generate response")

            logger.debug(f"Using max_tokens={safe_max_tokens} (model limit: 8192)")

            # Temporarily override max_tokens for this call
            temp_llm = self.llm.bind(max_tokens=safe_max_tokens)
            temp_chain = temp_llm.with_structured_output(GraphOutput)

            # Get structured output
            graph_output = await temp_chain.ainvoke([
                ("system", "You are a knowledge graph construction expert. Extract and refine entities and relationships from video analysis."),
                ("user", prompt)
            ])

            # Convert to dictionary format
            triplets = [
                {
                    "head": triplet.head,
                    "relation": triplet.relation,
                    "tail": triplet.tail
                }
                for triplet in graph_output.triplets
            ]

            logger.info(f"Extracted {len(triplets)} final triplets")
            # Write results to trace file
            if trace_file_path:
                try:
                    with open(trace_file_path, 'a', encoding='utf-8') as tf:
                        tf.write("\n===== FINAL LLM RESPONSE (TRIPLETS) =====\n")
                        tf.write(json.dumps(triplets, ensure_ascii=False, indent=2))
                        tf.write("\n===== END RESPONSE =====\n")
                except Exception as _:
                    pass
            logger.debug(f"Extracted triplets: {json.dumps(triplets[:5], indent=2)}...")

            return triplets

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            # Also write failure to trace file
            if trace_file_path:
                try:
                    with open(trace_file_path, 'a', encoding='utf-8') as tf:
                        tf.write("\n===== FINAL LLM ERROR =====\n")
                        tf.write(repr(e))
                        tf.write("\n===== END ERROR =====\n")
                except Exception as _:
                    pass
            if aggregated_content:
                logger.error(f"Content length: {len(aggregated_content)}")
            # Fallback: raw JSON-only ask and manual parsing
            try:
                fail_safe_system = (
                    "You are a strict JSON generator. Return ONLY a JSON object with key 'triplets' (array of objects with 'head','relation','tail'). "
                    "No text before or after JSON. If nothing to add, return {\"triplets\": []}."
                )
                raw_ai_msg = await self.llm.ainvoke([
                    ("system", fail_safe_system),
                    ("user", prompt)
                ])
                raw_text = getattr(raw_ai_msg, 'content', str(raw_ai_msg))

                if trace_file_path:
                    try:
                        with open(trace_file_path, 'a', encoding='utf-8') as tf:
                            tf.write("\n===== FINAL RAW LLM TEXT (FALLBACK) =====\n")
                            tf.write(raw_text)
                            tf.write("\n===== END RAW TEXT =====\n")
                    except Exception:
                        pass

                parsed = None
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    import re
                    match = re.search(r"\{[\s\S]*\}", raw_text)
                    if match:
                        try:
                            parsed = json.loads(match.group(0))
                        except Exception:
                            parsed = None

                if isinstance(parsed, dict) and isinstance(parsed.get("triplets"), list):
                    triplets = []
                    for t in parsed["triplets"]:
                        if not isinstance(t, dict):
                            continue
                        head = str(t.get("head", "")).strip()
                        rel = str(t.get("relation", "")).strip()
                        tail = str(t.get("tail", "")).strip()
                        if head and rel and tail:
                            triplets.append({"head": head, "relation": rel, "tail": tail})

                    if trace_file_path:
                        try:
                            with open(trace_file_path, 'a', encoding='utf-8') as tf:
                                tf.write("\n===== FINAL PARSED TRIPLETS (FALLBACK) =====\n")
                                tf.write(json.dumps(triplets, ensure_ascii=False, indent=2))
                                tf.write("\n===== END PARSED =====\n")
                        except Exception:
                            pass

                    logger.info(f"Parsed {len(triplets)} triplets from fallback raw JSON")
                    return triplets[:global_limit]
            except Exception as inner:
                logger.debug(f"Fallback raw parsing failed: {inner}")
            # Fallback to pre-extracted if available
            if pre_extracted_triplets:
                logger.warning(f"Falling back to {len(pre_extracted_triplets)} pre-extracted triplets")
                if trace_file_path:
                    try:
                        with open(trace_file_path, 'a', encoding='utf-8') as tf:
                            tf.write("\n===== FALLBACK PRE-EXTRACTED TRIPLETS =====\n")
                            tf.write(json.dumps(pre_extracted_triplets[:global_limit], ensure_ascii=False, indent=2))
                            tf.write("\n===== END FALLBACK =====\n")
                    except Exception as _:
                        pass
                return pre_extracted_triplets[:global_limit]
            return []