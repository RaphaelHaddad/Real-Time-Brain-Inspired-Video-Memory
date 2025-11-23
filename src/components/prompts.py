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

# OPTIMIZATION 3: Instruction-Based Global Refinement
# Why: Enables advanced operations for smarter graph building with subgraph context.
# Operations: new_triplets, inter_chunk_relations, merge_instructions, prune_instructions
LLM_INJECTOR_INSTRUCTION_PROMPT_TEMPLATE = """
Perform advanced knowledge graph refinement by strictly executing the required operations.

RULE: If CONTEXT SUBGRAPHS is empty or says "No subgraph context available", set "inter_chunk_relations", "merge_instructions", and "prune_instructions" to []. ONLY add to "new_triplets" from NEW CANDIDATES. DO NOT invent IDs, entities, or relations—stick to provided data.

### CONTEXT SUBGRAPHS (Relevant Existing Knowledge):
The following entities and relationships already exist in the graph. The IDs (e.g., '123_45') allow you to reference specific entities for linking and merging.
{subgraph_context}

### NEW CANDIDATES (From Current Extraction Batch):
{pre_extracted_triplets}

### STRICT OUTPUT FORMAT:
You MUST return a single JSON object containing these four keys.
The format for each key is STRICTLY defined below:

1. "new_triplets": List of new facts NOT present in the CONTEXT.
   - Format: [[Head, Relation, Tail, [SourceIndices]], ...]
2. "inter_chunk_relations": List of new relations connecting entities from NEW CANDIDATES to entities in CONTEXT SUBGRAPHS.
   - Format: [[NewHead, Relation, ExistingTail, [SourceIndices]], ...]
3. "merge_instructions": List of instructions to merge local duplicates into existing entities.
   - Format: [{{"local": LocalName, "existing": ExistingName, "existing_id": ExistingID}}, ...]
4. "prune_instructions": List of instructions to remove noise or refuted facts from CONTEXT.
   - Format: [{{"head": HeadName, "relation": RelationName, "tail": TailName, "source_id": SourceID}}, ...]

### EXAMPLES:

// Merging "Man" (new) into "Person" (existing, ID 42_1)
"merge_instructions": [
  {{"local": "Man", "existing": "Person", "existing_id": "42_1"}}
]

// Linking a new "Document" (new) to a pre-existing "Lab Table" (ID 1_3)
"inter_chunk_relations": [
  ["Document", "placed_on", "Lab Table", [0, 1]]
]

// Pruning an outdated relation (Person | Wore | Sweater from chunk 5_2)
"prune_instructions": [
  {{"head": "Person", "relation": "Wore", "tail": "Sweater", "source_id": "5_2"}}
]

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

def get_llm_injector_instruction_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(LLM_INJECTOR_INSTRUCTION_PROMPT_TEMPLATE)

def build_benchmark_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(BENCHMARK_ANSWER_PROMPT_TEMPLATE)

def build_benchmark_evaluation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(BENCHMARK_EVALUATION_PROMPT_TEMPLATE)
