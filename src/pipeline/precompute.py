"""
Pre-retrieval computation module for PageRank and CH3-L3 algorithms.
These precomputed scores enhance graph traversal during retrieval.
"""

import asyncio
import json
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from ..core.config import PreRetrievalComputationConfig, Neo4jConfig, KGConfig
from ..components.neo4j_handler import Neo4jHandler
from ..core.logger import get_logger

logger = get_logger(__name__)


class PrecomputeManager:
    """
    Manages pre-retrieval computation for PageRank and CH3-L3 algorithms.
    Stores precomputed scores as node properties in Neo4j.
    """

    def __init__(
        self,
        config: PreRetrievalComputationConfig,
        neo4j_handler: Neo4jHandler,
        graph_uuid: str
    ):
        self.config = config
        self.neo4j_handler = neo4j_handler
        self.graph_uuid = graph_uuid

    async def precompute_all(self, methods: List[str]) -> Dict[str, Any]:
        """
        Run precomputation for specified methods.
        
        Args:
            methods: List of methods to precompute ('page_rank', 'ch3_l3')
        
        Returns:
            Dictionary with status and statistics for each method
        """
        results = {}
        
        for method in methods:
            logger.info(f"Starting precomputation for method: {method}")
            start_time = time.perf_counter()
            
            if method == "page_rank":
                stats = await self.precompute_pagerank()
                results["page_rank"] = stats
            elif method == "ch3_l3":
                stats = await self.precompute_ch3_l3()
                results["ch3_l3"] = stats
            else:
                logger.warning(f"Unknown precomputation method: {method}")
                results[method] = {"status": "error", "message": f"Unknown method: {method}"}
                continue
            
            elapsed = time.perf_counter() - start_time
            results[method]["elapsed_seconds"] = elapsed
            logger.info(f"Completed {method} precomputation in {elapsed:.2f}s")
        
        return results

    async def precompute_pagerank(self) -> Dict[str, Any]:
        """
        Precompute Personalized PageRank scores for all entity nodes.
        For each entity, store top-K nearest nodes based on PPR scores.
        
        Uses hub sampling approach: for each node, compute PPR with that node
        as the teleport target, then store top-K neighbors.
        
        Returns:
            Statistics about the precomputation
        """
        logger.info("Starting PageRank precomputation")
        config = self.config.pagerank
        
        # Build NetworkX graph from Neo4j
        G = await self._build_networkx_graph()
        
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, skipping PageRank precomputation")
            return {"status": "skipped", "reason": "empty_graph"}
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Compute PPR for each node as seed
        ppr_scores = {}
        nodes = list(G.nodes())
        
        for node in tqdm(nodes, desc="Computing PPR scores"):
            # Personalized PageRank with single seed node
            personalization = {n: 0.0 for n in nodes}
            personalization[node] = 1.0
            
            try:
                scores = nx.pagerank(
                    G,
                    alpha=1 - config.alpha,  # NetworkX uses alpha as damping, we use it as teleport
                    personalization=personalization,
                    max_iter=config.max_steps
                )
                
                # Filter and sort scores
                filtered_scores = [
                    (n, s) for n, s in scores.items()
                    if n != node and s >= config.min_score
                ]
                filtered_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top-K
                ppr_scores[node] = filtered_scores[:config.top_k_per_node]
                
            except Exception as e:
                logger.warning(f"PPR computation failed for node {node}: {e}")
                # For disconnected components, fall back to direct neighbors only
                neighbors = list(G.neighbors(node))
                if neighbors:
                    # Assign equal scores to direct neighbors
                    score_per_neighbor = 1.0 / len(neighbors)
                    ppr_scores[node] = [(n, score_per_neighbor) for n in neighbors[:config.top_k_per_node]]
                else:
                    ppr_scores[node] = []
        
        # Store scores in Neo4j
        await self._store_pagerank_scores(ppr_scores)
        
        # Set precomputation flag
        await self.neo4j_handler.set_precomputation_flag(self.graph_uuid, "page_rank", True)
        
        return {
            "status": "success",
            "nodes_processed": len(ppr_scores),
            "total_scores_stored": sum(len(v) for v in ppr_scores.values()),
            "avg_neighbors_per_node": np.mean([len(v) for v in ppr_scores.values()])
        }

    async def precompute_ch3_l3(self) -> Dict[str, Any]:
        """
        Precompute CH3-L3 (Cannistraci-Hebb 3-hop Local) scores.
        
        For each node, compute CH3-L3 scores to nodes within 3 hops,
        storing top-K candidates with scores above threshold.
        
        Uses CAR (Cannistraci-Alanis-Ravasi) proxy for fast approximation:
        score = common_neighbors / sqrt(degree(u) * degree(v))
        
        Returns:
            Statistics about the precomputation
        """
        logger.info("Starting CH3-L3 precomputation")
        config = self.config.ch3_l3
        
        # Build NetworkX graph from Neo4j
        G = await self._build_networkx_graph()
        
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty, skipping CH3-L3 precomputation")
            return {"status": "skipped", "reason": "empty_graph"}
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Pre-compute 3-hop reachable sets for all nodes
        logger.info("Computing 3-hop reachable sets...")
        reachable_3hop = {}
        for node in tqdm(G.nodes(), desc="Computing reachable sets"):
            reachable_3hop[node] = set(
                nx.single_source_shortest_path_length(G, node, cutoff=3).keys()
            )
        
        # Compute CH3-L3 scores for all nodes
        ch3l3_scores = {}
        nodes = list(G.nodes())
        
        for i in tqdm(range(0, len(nodes), config.batch_size), desc="Computing CH3-L3 scores"):
            batch = nodes[i:i + config.batch_size]
            
            for u in batch:
                scores = []
                candidates = reachable_3hop[u] - {u}
                
                for v in candidates:
                    if u == v:
                        continue
                    
                    # Check if direct edge exists (score = 1.0)
                    if G.has_edge(u, v):
                        scores.append((v, 1.0))
                        continue
                    
                    # Compute CH3-L3 score using CAR proxy
                    score = self._fast_ch3l3(G, u, v, reachable_3hop, config.external_degree_approx)
                    
                    if score >= config.min_ch3_score:
                        scores.append((v, score))
                
                # Add direct neighbors with max score
                for neighbor in G.neighbors(u):
                    if not any(n == neighbor for n, _ in scores):
                        scores.append((neighbor, 1.0))
                
                # Sort and keep top-K
                scores.sort(key=lambda x: x[1], reverse=True)
                ch3l3_scores[u] = scores[:config.candidates_per_node]
        
        # Store scores in Neo4j
        await self._store_ch3l3_scores(ch3l3_scores)
        
        # Set precomputation flag
        await self.neo4j_handler.set_precomputation_flag(self.graph_uuid, "ch3_l3", True)
        
        return {
            "status": "success",
            "nodes_processed": len(ch3l3_scores),
            "total_scores_stored": sum(len(v) for v in ch3l3_scores.values()),
            "avg_candidates_per_node": np.mean([len(v) for v in ch3l3_scores.values()])
        }

    def _fast_ch3l3(
        self,
        G: nx.Graph,
        u: str,
        v: str,
        reachable_cache: Dict[str, Set[str]],
        external_degree_approx: int = 2
    ) -> float:
        """
        Fast CH3-L3 score approximation using CAR (Cannistraci-Alanis-Ravasi) proxy.
        
        Args:
            G: NetworkX graph
            u: Source node
            v: Target node
            reachable_cache: Pre-computed 3-hop reachable sets
            external_degree_approx: Approximation factor for external degree
        
        Returns:
            CH3-L3 score (0.0 if no connection)
        """
        # Find common 3-hop reachable nodes
        common = reachable_cache[u] & reachable_cache[v]
        
        if len(common) < external_degree_approx:
            return 0.0
        
        # Common neighbors (1-hop)
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        cn = len(u_neighbors & v_neighbors)
        
        if cn == 0:
            return 0.0
        
        # CAR proxy: cn / sqrt(degree(u) * degree(v))
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        
        if deg_u * deg_v == 0:
            return 0.0
        
        return cn / np.sqrt(deg_u * deg_v)

    async def _build_networkx_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from Neo4j entities and relationships.
        
        Returns:
            NetworkX Graph with entity names as nodes
        """
        G = nx.Graph()
        
        async with self.neo4j_handler.driver.session() as session:
            # Get all entities
            result = await session.run(
                """
                MATCH (e:Entity {graph_uuid: $graph_uuid})
                RETURN e.name AS name
                """,
                graph_uuid=self.graph_uuid
            )
            
            async for record in result:
                G.add_node(record["name"])
            
            # Get all relationships between entities
            result = await session.run(
                """
                MATCH (e1:Entity {graph_uuid: $graph_uuid})-[r]->(e2:Entity {graph_uuid: $graph_uuid})
                WHERE type(r) <> 'FROM_CHUNK'
                RETURN DISTINCT e1.name AS source, e2.name AS target, type(r) AS rel_type
                """,
                graph_uuid=self.graph_uuid
            )
            
            async for record in result:
                G.add_edge(record["source"], record["target"], rel_type=record["rel_type"])
        
        return G

    async def _store_pagerank_scores(self, ppr_scores: Dict[str, List[Tuple[str, float]]]) -> None:
        """
        Store PageRank scores as node properties in Neo4j.
        
        Args:
            ppr_scores: Dictionary mapping node names to list of (neighbor, score) tuples
        """
        logger.info(f"Storing PageRank scores for {len(ppr_scores)} nodes")
        
        async with self.neo4j_handler.driver.session() as session:
            for node_name, scores in tqdm(ppr_scores.items(), desc="Storing PPR scores"):
                # Convert scores to JSON string
                scores_json = json.dumps([
                    {"node": n, "score": float(s)} for n, s in scores
                ])
                
                await session.run(
                    """
                    MATCH (e:Entity {name: $name, graph_uuid: $graph_uuid})
                    SET e.ppr_top_neighbors = $scores
                    """,
                    name=node_name,
                    graph_uuid=self.graph_uuid,
                    scores=scores_json
                )

    async def _store_ch3l3_scores(self, ch3l3_scores: Dict[str, List[Tuple[str, float]]]) -> None:
        """
        Store CH3-L3 scores as node properties in Neo4j.
        
        Args:
            ch3l3_scores: Dictionary mapping node names to list of (candidate, score) tuples
        """
        logger.info(f"Storing CH3-L3 scores for {len(ch3l3_scores)} nodes")
        
        async with self.neo4j_handler.driver.session() as session:
            for node_name, scores in tqdm(ch3l3_scores.items(), desc="Storing CH3-L3 scores"):
                # Convert scores to JSON string
                scores_json = json.dumps([
                    {"node": n, "score": float(s)} for n, s in scores
                ])
                
                await session.run(
                    """
                    MATCH (e:Entity {name: $name, graph_uuid: $graph_uuid})
                    SET e.ch3l3_scores = $scores
                    """,
                    name=node_name,
                    graph_uuid=self.graph_uuid,
                    scores=scores_json
                )


async def run_precomputation(
    config_path: str,
    graph_uuid: str,
    methods: List[str],
    param_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run precomputation for a graph with given methods.
    
    Args:
        config_path: Path to configuration file
        graph_uuid: UUID of the graph to precompute
        methods: List of methods ('page_rank', 'ch3_l3')
        param_overrides: Optional parameter overrides
    
    Returns:
        Results dictionary with statistics for each method
    """
    from ..core.config import PipelineConfig
    
    logger.info(f"Loading configuration from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)
    
    # Apply parameter overrides if provided
    if param_overrides:
        precompute_config = config.pre_retrieval_computation
        if "pagerank" in param_overrides:
            for k, v in param_overrides["pagerank"].items():
                setattr(precompute_config.pagerank, k, v)
        if "ch3_l3" in param_overrides:
            for k, v in param_overrides["ch3_l3"].items():
                setattr(precompute_config.ch3_l3, k, v)
    
    logger.info(f"Running precomputation for graph: {graph_uuid}")
    logger.info(f"Methods: {methods}")
    
    # Initialize Neo4j handler
    neo4j_handler = Neo4jHandler(config.neo4j, config.kg, graph_uuid)
    
    try:
        # Check if graph exists
        exists = await neo4j_handler.graph_exists(graph_uuid)
        if not exists:
            raise ValueError(f"Graph with UUID {graph_uuid} does not exist")
        
        # Run precomputation
        manager = PrecomputeManager(
            config.pre_retrieval_computation,
            neo4j_handler,
            graph_uuid
        )
        
        results = await manager.precompute_all(methods)
        
        return results
    
    finally:
        await neo4j_handler.close()
