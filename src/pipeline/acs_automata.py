from ..core.logger import get_logger
from ..components.neo4j_handler import Neo4jHandler
from typing import Dict, Any
import time
import networkx as nx
import math

logger = get_logger(__name__)

class ACSAutomata:
    """
    ACS (Adaptive Computing System) Automata
    Computes network science metrics using NetworkX (in-memory) for reliability and simplicity.
    """
    
    def __init__(self, neo4j_handler: Neo4jHandler):
        self.neo4j_handler = neo4j_handler
        self.metrics_cache = {}
        self.last_update_time = time.time()
        
    async def update_metrics(self) -> Dict[str, Any]:
        """Update and compute network science metrics using NetworkX"""
        start_time = time.perf_counter()
        
        try:
            logger.info("Computing network science metrics using NetworkX...")
            
            # Build NetworkX graph from Neo4j
            G = await self._build_networkx_graph()
            
            # Basic metrics
            node_count = G.number_of_nodes()
            rel_count = G.number_of_edges()
            
            # Density
            if node_count < 2:
                density = 0.0
            else:
                density = nx.density(G)
            
            # Avg Degree (total edges)
            if node_count == 0:
                avg_degree = 0.0
            else:
                # For directed graph: sum(in_degree + out_degree) / n = 2 * m / n
                avg_degree = (2 * rel_count) / node_count
            
            # Avg Unique Neighbors Degree (unique connections, ignoring multi-edges)
            if node_count == 0:
                avg_unique_neighbors = 0.0
            else:
                # Convert to undirected to count unique neighbors only
                G_undir = G.to_undirected()
                # Remove self-loops if any
                G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
                degrees = dict(G_undir.degree())
                avg_unique_neighbors = sum(degrees.values()) / len(degrees) if degrees else 0.0
            
            # Global Efficiency (shows inter-chunk connectivity benefits)
            # Compute on undirected largest connected component for meaningful values.
            try:
                G_undir = G.to_undirected()
                if node_count < 2:
                    global_efficiency = 0.0
                else:
                    # For small graphs compute exact; for large graphs compute sampling approximation
                    SMALL_LIMIT = 500
                    if node_count <= SMALL_LIMIT:
                        global_efficiency = nx.global_efficiency(G_undir)
                        logger.debug(f"Computed exact global_efficiency on {node_count} nodes")
                    else:
                        # Sampling-based approximation for global efficiency
                        import random
                        random.seed(42)
                        SAMPLE_SOURCES = min(100, node_count)
                        nodes = list(G_undir.nodes())
                        sample_sum = 0.0
                        sample_count = 0
                        for _ in range(SAMPLE_SOURCES):
                            source = random.choice(nodes)
                            lengths = nx.single_source_shortest_path_length(G_undir, source)
                            # count reachable distances as reciprocal contributions
                            for target, d in lengths.items():
                                if target == source:
                                    continue
                                if d and d > 0:
                                    sample_sum += 1.0 / d
                            # Include unreachable nodes as zero contributions: count total possible targets
                            sample_count += (len(nodes) - 1)
                        global_efficiency = (sample_sum / sample_count) if sample_count > 0 else 0.0
                        logger.debug(f"Computed approximate global_efficiency with {SAMPLE_SOURCES} samples")
            except Exception:
                global_efficiency = 0.0
            
            # Average Path Length (shorter paths = better merging)
            try:
                # Compute path lengths on undirected largest connected component for meaningful average distances
                G_undir = G.to_undirected()
                if G_undir.number_of_nodes() < 2:
                    avg_path_length = None
                else:
                    connected = nx.is_connected(G_undir)
                    if connected:
                        comp = G_undir
                    else:
                        largest_cc = max(nx.connected_components(G_undir), key=len)
                        comp = G_undir.subgraph(largest_cc)

                    # For small graphs compute exact; else sample nodes to approximate average
                    SMALL_LIMIT = 500
                    if comp.number_of_nodes() <= SMALL_LIMIT:
                        avg_path_length = nx.average_shortest_path_length(comp)
                        logger.debug(f"Computed exact avg_path_length on component with {comp.number_of_nodes()} nodes")
                    else:
                        # Sampling-based approximation of average shortest path length
                        import random
                        random.seed(42)
                        nodes = list(comp.nodes())
                        SAMPLE_SOURCES = min(100, len(nodes))
                        total_sum = 0.0
                        total_count = 0
                        for _ in range(SAMPLE_SOURCES):
                            source = random.choice(nodes)
                            lengths = nx.single_source_shortest_path_length(comp, source)
                            for target, d in lengths.items():
                                if target == source:
                                    continue
                                total_sum += d
                                total_count += 1
                        avg_path_length = (total_sum / total_count) if total_count > 0 else None
                        logger.debug(f"Computed approximate avg_path_length with {SAMPLE_SOURCES} samples on component size {len(nodes)}")
            except Exception:
                avg_path_length = None
            
            # Degree Centrality (shows entity importance after merging)
            try:
                degree_centrality = nx.degree_centrality(G)
                avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
                max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0.0
            except Exception:
                avg_degree_centrality = 0.0
                max_degree_centrality = 0.0
            
            # Betweenness Centrality (shows bridging entities from inter-chunk)
            try:
                betweenness = nx.betweenness_centrality(G, k=min(100, node_count))  # Approximate for large graphs
                avg_betweenness = sum(betweenness.values()) / len(betweenness)
                max_betweenness = max(betweenness.values()) if betweenness else 0.0
            except Exception:
                avg_betweenness = 0.0
                max_betweenness = 0.0
            
            # Assortativity (shows if similar entities connect - benefit of merging)
            try:
                degree_assortativity = nx.degree_assortativity_coefficient(G)
            except Exception:
                degree_assortativity = 0.0
            
            # Graph Robustness (random node removal)
            try:
                # Simulate removing 10% of nodes randomly
                if node_count > 10:
                    nodes_to_remove = int(0.1 * node_count)
                    G_robust = G.copy()
                    import random
                    nodes_list = list(G_robust.nodes())
                    random.seed(42)  # For reproducibility
                    nodes_to_remove_list = random.sample(nodes_list, min(nodes_to_remove, len(nodes_list)))
                    G_robust.remove_nodes_from(nodes_to_remove_list)
                    
                    if G_robust.number_of_nodes() > 1:
                        robustness = G_robust.number_of_edges() / G.number_of_edges()
                    else:
                        robustness = 0.0
                else:
                    robustness = 1.0
            except Exception:
                robustness = 0.0
            
            # Diameter (estimate)
            diameter_estimate = 0
            if node_count > 0:
                 if node_count < 500: # Only compute for small graphs
                    try:
                        # Diameter is defined for connected graphs. Use largest WCC.
                        if nx.is_weakly_connected(G):
                             diameter_estimate = nx.diameter(G.to_undirected())
                        else:
                             # Get largest component
                             largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                             subgraph = G.to_undirected().subgraph(largest_cc)
                             diameter_estimate = nx.diameter(subgraph)
                    except Exception:
                        diameter_estimate = min(node_count // 2, 50)
                 else:
                    diameter_estimate = min(node_count // 2, 50)

            # Clustering Coefficient (Average)
            try:
                clustering_coefficient = nx.average_clustering(G)
            except Exception:
                clustering_coefficient = 0.0

            # Weakly Connected Components
            wcc = list(nx.weakly_connected_components(G))
            wcc_count = len(wcc)
            largest_wcc_size = max(len(c) for c in wcc) if wcc else 0

            # PageRank
            try:
                pagerank = nx.pagerank(G)
                scores = list(pagerank.values())
                total_pr = sum(scores)
                scores.sort(reverse=True)
                top10 = scores[:10]
                pagerank_top10_percent = (sum(top10) / total_pr * 100.0) if total_pr > 0 else 0.0
            except Exception:
                pagerank_top10_percent = 0.0

            # Louvain Communities
            louvain_communities = 0
            louvain_modularity = 0.0
            try:
                G_undir = G.to_undirected()
                # Check if louvain is available in nx (v2.8+)
                if hasattr(nx.community, 'louvain_communities'):
                    comms = nx.community.louvain_communities(G_undir)
                    louvain_communities = len(comms)
                    louvain_modularity = nx.community.modularity(G_undir, comms)
                else:
                    logger.warning("nx.community.louvain_communities not available")
            except Exception as e:
                logger.warning(f"Louvain computation failed: {e}")

            # Label Entropy
            label_entropy = await self._compute_label_entropy()

            metrics = {
                "node_count": node_count,
                "relationship_count": rel_count,
                "density": round(density, 4),
                "avg_degree": round(avg_degree, 4),
                "avg_unique_neighbors": round(avg_unique_neighbors, 4),
                "global_efficiency": round(global_efficiency, 4),
                "avg_path_length": round(avg_path_length, 4) if avg_path_length is not None else None,
                "avg_degree_centrality": round(avg_degree_centrality, 4),
                "max_degree_centrality": round(max_degree_centrality, 4),
                "avg_betweenness_centrality": round(avg_betweenness, 4),
                "max_betweenness_centrality": round(max_betweenness, 4),
                "degree_assortativity": round(degree_assortativity, 4),
                "graph_robustness": round(robustness, 4),
                "diameter_estimate": diameter_estimate,
                "clustering_coefficient": round(clustering_coefficient, 4),
                "weakly_connected_components": wcc_count,
                "largest_wcc_size": largest_wcc_size,
                "pagerank_top10_percent": round(pagerank_top10_percent, 2),
                "louvain_communities": louvain_communities,
                "louvain_modularity": round(louvain_modularity, 4) if louvain_modularity is not None else None,
                "label_entropy": label_entropy,
                "computational_time": time.perf_counter() - start_time
            }

            self.metrics_cache.update(metrics)
            self.last_update_time = time.time()
            
            logger.info(f"Network metrics computed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing network metrics: {str(e)}")
            return {"error": str(e), "computational_time": time.perf_counter() - start_time}

    async def _build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX graph from the Neo4j subgraph for the current run"""
        G = nx.DiGraph()
        try:
            async with self.neo4j_handler.driver.session() as session:
                # Fetch nodes filtered by graph_uuid
                node_query = """
                MATCH (n:GraphNode) 
                WHERE n.graph_uuid = $graph_uuid 
                RETURN elementId(n) as id
                """
                result_nodes = await session.run(node_query, graph_uuid=self.neo4j_handler.run_uuid)
                nodes = [record["id"] async for record in result_nodes]
                G.add_nodes_from(nodes)
                
                # Fetch edges where both source and target are in the graph_uuid
                edge_query = """
                MATCH (n:GraphNode)-[r]->(m:GraphNode)
                WHERE n.graph_uuid = $graph_uuid AND m.graph_uuid = $graph_uuid
                RETURN elementId(n) as source, elementId(m) as target
                """
                result_edges = await session.run(edge_query, graph_uuid=self.neo4j_handler.run_uuid)
                edges = [(record["source"], record["target"]) async for record in result_edges]
                G.add_edges_from(edges)
                
            return G
        except Exception as e:
            logger.error(f"Error building NetworkX graph: {str(e)}")
            return nx.DiGraph()

    async def _compute_label_entropy(self) -> float:
        """Compute Shannon entropy of node labels"""
        try:
            async with self.neo4j_handler.driver.session() as session:
                lbl_q = """
                MATCH (n:GraphNode) WHERE n.graph_uuid = $graph_uuid
                UNWIND labels(n) as lab
                RETURN lab, count(*) as cnt
                """
                res = await session.run(lbl_q, graph_uuid=self.neo4j_handler.run_uuid)
                label_counts = {rec['lab']: rec['cnt'] async for rec in res}
                
                total = sum(label_counts.values())
                entropy = 0.0
                if total > 0:
                    for c in label_counts.values():
                        p = c / total
                        entropy -= p * math.log(p, 2)
                return round(entropy, 4)
        except Exception as e:
            logger.error(f"Failed to compute label entropy: {str(e)}")
            return 0.0