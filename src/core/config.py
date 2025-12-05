from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
import yaml


class PageRankConfig(BaseModel):
    """Configuration for PageRank precomputation"""
    alpha: float = 0.15  # Teleport probability (damping factor = 1-alpha)
    max_steps: int = 20  # Max iterations for power iteration
    top_k_per_node: int = 50  # Top K neighbors to store per node
    min_score: float = 0.001  # Minimum score threshold to store


class CH3L3Config(BaseModel):
    """Configuration for CH3-L3 precomputation"""
    candidates_per_node: int = 200  # Top K candidates to store per node
    min_ch3_score: float = 0.2  # Minimum CH3-L3 score threshold
    external_degree_approx: int = 2  # Approximation factor for external degree
    batch_size: int = 50  # Batch size for processing nodes


class PreRetrievalComputationConfig(BaseModel):
    """Configuration for pre-retrieval computation (PageRank, CH3-L3)"""
    auto_precompute_on_kg_build: bool = False  # Run precomputation automatically after KG build
    pagerank: PageRankConfig = PageRankConfig()
    ch3_l3: CH3L3Config = CH3L3Config()


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
    use_reranker: bool = False
    reranker_endpoint: Optional[str] = None
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    top_k: int = 5
    top_k_chunks: int = 8  # Initial vector search on chunks
    top_k_entities: int = 5  # Parallel search on entities
    top_k_relationships: int = 5  # Limit for displayed relationships
    graph_hops: int = 2
    post_compression: bool = True  # Enable contextual compression
    compression_threshold: float = 0.7  # Minimum similarity for retained chunks
    verbose: bool = False
    entity_first: bool = False  # Skip vector search, rely on entities + traversal
    rerank_after_traversal: bool = False  # Rerank after traversal instead of after vector search
    rerank_entities: bool = False  # Rerank entities separately
    rerank_relationships: bool = False  # Rerank relationships separately
    # Hop method: 'naive' (default BFS), 'page_rank' (PPR-guided), 'ch3_l3' (path-based)
    hop_method: Literal["naive", "page_rank", "ch3_l3"] = "naive"
    # CH3-L3 specific retrieval parameters
    top_k_hop_ch3_l3: int = 3  # Top K candidates per hop for CH3-L3
    max_path_length_ch3: int = 3  # Maximum path length for CH3-L3 traversal
    cum_score_aggregation: Literal["product", "sum"] = "product"  # Score aggregation method
    # PageRank specific retrieval parameters
    top_k_hop_pagerank: int = 10  # Top K candidates per hop for PageRank

class BenchmarkLLMConfig(BaseModel):
    """Configuration for benchmark evaluation using LLM"""
    endpoint: str
    api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048

class CommunityHighGraphConfig(BaseModel):
    """Configuration for community-based high-level graph construction and retrieval"""
    community_creator: bool = False  # Enable community-aware KG building with question generation
    community_retriever: bool = False  # Enable community-based retrieval (similarity on CommunitySummary nodes)
    question_per_chunk: int = 2  # Number of questions to generate per chunk
    frequency_incremental_leiden: int = 5  # Run Leiden community detection every N batches
    community_resolution: float = 1.0  # Gamma parameter for Leiden algorithm resolution
    debug_print_community: int = 3  # Number of random community descriptions to log at DEBUG level

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
    community_high_graph: CommunityHighGraphConfig = CommunityHighGraphConfig()
    pre_retrieval_computation: PreRetrievalComputationConfig = PreRetrievalComputationConfig()
    # Save batch-level network metrics to a separate file after each batch
    saving_batch_metrics: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.parse_obj(data)