from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import yaml

class VideoConfig(BaseModel):
    chunk_size_seconds: float = Field(5.0, gt=0)
    frames_per_chunk: int = Field(5, gt=0)
    fps_target: Optional[int] = None

class VLMConfig(BaseModel):
    endpoint: str
    api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful assistant that describes video content in detail."
    user_prompt_template: str = "Describe what's happening in these frames from a video."

class LLMInjectorConfig(BaseModel):
    endpoint: str
    api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    subgraph_extraction_injection: bool = False

class ChunkingConfig(BaseModel):
    """Configuration for hierarchical chunking strategy"""
    enabled: bool = True
    chunk_size: int = 320  # tokens (smaller for faster processing)
    chunk_overlap: int = 64  # tokens
    max_triplets_per_chunk: int = 6  # Local extraction limit (keep it small)
    use_sentence_boundaries: bool = True  # Respect narrative flow
    parallel_count: int = 4  # Concurrent chunk extractions
    enable_global_refinement: bool = True  # Merge & refine triplets
    refinement_max_tokens: int = 500  # Global refinement call limit (reduced)
    global_triplet_limit: int = 25  # Max triplets passed to refinement/final LLM
    batch_llm_parallelism: bool = True  # Enable parallel LLM calls; set to false for serial execution
    # Limits for instruction-based global refinement outputs
    max_new_triplets: int = 15
    max_inter_chunk_relations: int = 15
    max_merge_instructions: int = 10
    max_prune_instructions: int = 10
    # Timeout & retry for per-chunk LLM extraction
    chunk_timeout_seconds: float = 45.0
    chunk_timeout_retries: int = 0

class KGConfig(BaseModel):
    batch_size: int = Field(6, gt=0)
    verbose: bool = False
    embedding_endpoint: str
    embedding_model: str
    embedding_api_key: str

class EmbedderConfig(BaseModel):
    endpoint: str
    api_key: str
    model: str
    top_k_chunk_with_batch_similarity: int = 3
    top_k_similar_batch: int = 2

class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str
    database: str = "neo4j"

class RetrievalConfig(BaseModel):
    endpoint: str
    api_key: str
    model_name: str
    use_reranker: bool = False
    reranker_endpoint: Optional[str] = None
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    top_k: int = 5
    top_k_chunks: int = 8  # Initial vector search on chunks
    top_k_entities: int = 5  # Parallel search on entities
    graph_hops: int = 2
    post_compression: bool = True  # Enable contextual compression
    compression_threshold: float = 0.7  # Minimum similarity for retained chunks
    verbose: bool = False

class BenchmarkLLMConfig(BaseModel):
    """Configuration for benchmark evaluation using LLM"""
    endpoint: str
    api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048

class PipelineConfig(BaseModel):
    video: VideoConfig
    vlm: VLMConfig
    llm_injector: LLMInjectorConfig
    chunking: ChunkingConfig = ChunkingConfig()
    kg: KGConfig
    embedder: EmbedderConfig
    neo4j: Neo4jConfig
    retrieval: RetrievalConfig
    benchmark_llm: Optional[BenchmarkLLMConfig] = None
    # Save batch-level network metrics to a separate file after each batch
    saving_batch_metrics: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.parse_obj(data)