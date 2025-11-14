"""
Centralized prompt templates for VidGraph components.

Provides:
- PRE_LLM_PROMPT_TEMPLATE (raw string with {max_triplets} and {input} placeholders)
- LLM_INJECTOR_PROMPT_TEMPLATE (raw string with {network_info} and {pre_extracted_triplets})
- helper builders to create ChatPromptTemplate instances when needed

Keep templates here to allow easy editing without touching business logic.
"""
from langchain_core.prompts import ChatPromptTemplate


PRE_LLM_PROMPT_TEMPLATE = """
Extract key entities and relationships from this video description chunk.

INSTRUCTIONS:
- Focus on concrete, directly observable entities: people, objects, actions, locations.
- Identify ONLY direct observable relationships between entities (no speculation).
- Return AT MOST {max_triplets} triplets (prioritize most important).
- Be concise and precise.

OUTPUT FORMAT (MANDATORY):
Return a JSON array (no surrounding text) where each element is an object with keys
"head", "relation", "tail". Example (literal braces shown):

[{{"head": "Person", "relation": "holds", "tail": "Smartphone"}},
 {{"head": "Table", "relation": "contains", "tail": "Beaker"}}]

Now extract the triplets from the following chunk. Return ONLY the JSON array.

Chunk:
{input}
"""


LLM_INJECTOR_PROMPT_TEMPLATE = """
You are a knowledge graph construction expert. Your task is to CONSOLIDATE and ENRICH a set of candidate triplets.

NETWORK CONTEXT:
{network_info}

CANDIDATE TRIPLETS (from hierarchical local extraction):
{pre_extracted_triplets}

INSTRUCTIONS:
1. Normalize entity names (consistent casing, merge aliases)
2. Merge exact or near-duplicate triplets (same semantic meaning)
3. Add ONLY high-value missing relations directly implied by the set
4. Do NOT re-read or reference original raw text; rely solely on provided triplets + network context
5. Use snake_case for relation names
6. Limit output to essential relations (avoid speculative or redundant links)
7. For each output triplet, include "source_chunks" array listing the original chunk indices where this triplet was found or derived from

STRICT OUTPUT CONTRACT:
- Return ONLY a single JSON object with exactly one top-level key: "triplets" (an array)
- Each array element must be an object with keys: "head" (string), "relation" (string), "tail" (string), "source_chunks" (array of integers)
- Do not include any other keys or any text outside the JSON object
- If there are no triplets to return, output: {"triplets": []}

Example:
{"triplets": [{"head": "Person", "relation": "holds", "tail": "Smartphone", "source_chunks": [0, 1]}, {"head": "Smartphone", "relation": "brand", "tail": "Apple", "source_chunks": [2]}]}
"""


def build_pre_llm_prompt_template(max_triplets: int) -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for pre-LLM chunk extraction with max_triplets injected."""
    # Keep the {input} placeholder intact for ChatPromptTemplate usage
    text = PRE_LLM_PROMPT_TEMPLATE.replace("{max_triplets}", str(max_triplets))
    return ChatPromptTemplate.from_template(text)


def get_llm_injector_prompt_template() -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for the final LLM injector."""
    return ChatPromptTemplate.from_template(LLM_INJECTOR_PROMPT_TEMPLATE)


BENCHMARK_ANSWER_PROMPT_TEMPLATE = """
You are an expert at answering questions based on provided context from video analysis.

QUESTION:
{question}

CONTEXT FROM VIDEO RETRIEVAL:
{context}

INSTRUCTIONS:
1. Use ONLY the provided context to answer the question
2. If the context doesn't contain enough information to answer, state that clearly
3. Be concise and direct in your answer
4. Base your answer entirely on what is mentioned in the context

Provide your answer:
"""


BENCHMARK_EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator assessing whether an AI-generated answer correctly answers a question based on provided context.

QUESTION:
{question}

GROUND TRUTH (expected answer):
{groundtruth}

AI-GENERATED ANSWER:
{generated_answer}

CONTEXT USED:
{context}

INSTRUCTIONS:
1. Evaluate if the generated answer correctly answers the question
2. Consider semantic equivalence (the answer doesn't need to be word-for-word identical)
3. Check if the answer is supported by the provided context
4. Return ONLY a JSON object with exactly one key "is_correct" with boolean value

OUTPUT FORMAT (MANDATORY):
Return ONLY a JSON object with this exact structure:
{{"is_correct": true}}
or
{{"is_correct": false}}

Evaluation:
"""


def build_benchmark_answer_prompt() -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for benchmark answer generation."""
    return ChatPromptTemplate.from_template(BENCHMARK_ANSWER_PROMPT_TEMPLATE)


def build_benchmark_evaluation_prompt() -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for benchmark answer evaluation."""
    return ChatPromptTemplate.from_template(BENCHMARK_EVALUATION_PROMPT_TEMPLATE)
