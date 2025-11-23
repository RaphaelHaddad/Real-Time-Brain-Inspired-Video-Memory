"""
Centralized prompt templates for VidGraph components.

OPTIMIZATIONS:
- PRE_LLM: Switched to Pipe-delimited output ("Head | Relation | Tail") to reduce token generation.
- LLM_INJECTOR: Switched to JSON List-of-Lists format to reduce structure overhead while maintaining strict parsing for source tracking.
"""
from langchain_core.prompts import ChatPromptTemplate

# OPTIMIZATION 1: Pipe-Delimited Output
# Why: Removes JSON overhead (braces, quotes, keys) for faster generation.
# Parsing: String.split('|') is faster and more robust than json.loads() for simple triplets.
PRE_LLM_PROMPT_TEMPLATE = """
Extract observable entities and direct relationships from the video text below.

STRICT FORMAT:
- Output ONE triplet per line.
- Use the format: Entity1 | relation_in_snake_case | Entity2
- Do not use quotes or numbering.
- Max {max_triplets} lines.

EXAMPLE:
Person | holds | Smartphone
Lab Table | contains | Beaker

INPUT TEXT:
{input}

OUTPUT:
"""

# OPTIMIZATION 2: Compact JSON Arrays
# Why: Retains structural integrity for complex merging/source tracking but removes repetitive keys.
# Format: [Head, Relation, Tail, [SourceIndices]]
LLM_INJECTOR_PROMPT_TEMPLATE = """
Consolidate and enrich the following knowledge graph triplets.

CONTEXT:
{network_info}

CANDIDATES:
{pre_extracted_triplets}

TASKS:
1. Merge duplicates and normalize entities (e.g., "Man" -> "Person").
2. Use snake_case for relations.
3. Track source chunks indices.

OUTPUT FORMAT:
Return a JSON object with a single key "triplets".
The value must be a list of lists, where each inner list is: [Head, Relation, Tail, [source_indices]].

EXAMPLE:
{{"triplets": [
  ["Person", "holds", "Smartphone", [0, 1]],
  ["Smartphone", "brand", "Apple", [2]]
]}}

JSON OUTPUT:
"""

# Benchmark prompts remain largely the same as they rely on natural language generation
BENCHMARK_ANSWER_PROMPT_TEMPLATE = """
Answer the question based ONLY on the video context provided.

QUESTION: {question}

CONTEXT:
{context}

ANSWER (Concise):
"""

BENCHMARK_EVALUATION_PROMPT_TEMPLATE = """
Evaluate if the AI Answer matches the Ground Truth based on the Context.

QUESTION: {question}
GROUND TRUTH: {groundtruth}
AI ANSWER: {generated_answer}
CONTEXT: {context}

Return strictly JSON: {{"is_correct": true}} or {{"is_correct": false}}
"""

def build_pre_llm_prompt_template(max_triplets: int) -> ChatPromptTemplate:
    text = PRE_LLM_PROMPT_TEMPLATE.replace("{max_triplets}", str(max_triplets))
    return ChatPromptTemplate.from_template(text)

def get_llm_injector_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(LLM_INJECTOR_PROMPT_TEMPLATE)

def build_benchmark_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(BENCHMARK_ANSWER_PROMPT_TEMPLATE)

def build_benchmark_evaluation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(BENCHMARK_EVALUATION_PROMPT_TEMPLATE)
