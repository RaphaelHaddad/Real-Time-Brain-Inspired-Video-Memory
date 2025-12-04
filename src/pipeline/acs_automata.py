from ..core.logger import get_logger
from ..components.neo4j_handler import Neo4jHandler
from typing import Dict, Any, List, Tuple, Set
import time
import networkx as nx
import math
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)

class ACSAutomata:
    """
    ACS (Adaptive Computing System) Automata
    Computes network science metrics using NetworkX (in-memory).
    Includes semantic pruning of redundant edges using embeddings to optimize graph structure.
    """
    
    def __init__(self, neo4j_handler: Neo4jHandler):
        self.neo4j_handler = neo4j_handler
        self.metrics_cache = {}
        self.last_update_time = time.time()
        
        # Initialize embedding model for semantic pruning
        try:
            logger.info("Loading embedding model for graph pruning (Qwen/Qwen3-Embedding-0.6B)...")
            self.embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
    async def update_metrics(self) -> Dict[str, Any]:
        """Update and compute network science metrics using NetworkX"""
        start_time = time.perf_counter()
        
        try:
            logger.info("Computing network science metrics using NetworkX...")
            
            # Build NetworkX MultiDiGraph (to capture all parallel edges for pruning)
            G_multi = await self._build_networkx_graph()
            
            # Prune redundant edges if model is available
            if self.embedding_model and G_multi.number_of_edges() > 0:
                G_pruned = self._prune_redundant_edges(G_multi)
            else:
                G_pruned = G_multi

            # Convert to simple DiGraph for standard metrics 
            # (This collapses any remaining parallel edges to single edges, which is standard for these metrics)
            G = nx.DiGraph(G_pruned)
            
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
            top_central_nodes = []
            try:
                degree_centrality = nx.degree_centrality(G)
                avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
                max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0.0
                
                # Capture top nodes for LLM context
                sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                top_central_nodes = [
                    {"name": G.nodes[n_id].get("name", str(n_id)), "score": score} 
                    for n_id, score in sorted_centrality
                ]
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

            # --- Semantic Metrics ---
            # Relation type distribution
            edge_types = [d.get("type") for _, _, d in G.edges(data=True) if "type" in d]
            relation_counts = Counter(edge_types)
            top_relations = [{"type": k, "count": v} for k, v in relation_counts.most_common(10)]
            
            # Entity label distribution
            node_labels = []
            for _, d in G.nodes(data=True):
                if "labels" in d:
                    node_labels.extend(d["labels"])
            label_counts = Counter(node_labels)
            top_entity_types = [{"label": k, "count": v} for k, v in label_counts.most_common(10)]

            # --- Growth Indicators ---
            hub_concentration = 0.0
            if node_count > 0:
                degrees = [d for _, d in G.degree()]
                top_10_percent = max(1, int(0.1 * node_count))
                top_degrees = sorted(degrees, reverse=True)[:top_10_percent]
                total_degree_sum = sum(degrees)
                if total_degree_sum > 0:
                    hub_concentration = sum(top_degrees) / total_degree_sum

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
                "top_central_nodes": top_central_nodes,
                "top_relations": top_relations,
                "top_entity_types": top_entity_types,
                "hub_concentration": round(hub_concentration, 4),
                "computational_time": time.perf_counter() - start_time
            }

            self.metrics_cache.update(metrics)
            self.last_update_time = time.time()
            
            logger.info(f"Network metrics computed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing network metrics: {str(e)}")
            return {"error": str(e), "computational_time": time.perf_counter() - start_time}

    async def _build_networkx_graph(self) -> nx.MultiDiGraph:
        """Build a NetworkX MultiDiGraph from the Neo4j subgraph to capture all edges"""
        G = nx.MultiDiGraph()
        try:
            async with self.neo4j_handler.driver.session() as session:
                # Fetch nodes filtered by graph_uuid with name and labels
                node_query = """
                MATCH (n:GraphNode) 
                WHERE n.graph_uuid = $graph_uuid 
                RETURN elementId(n) as id, n.name as name, labels(n) as labels
                """
                result_nodes = await session.run(node_query, graph_uuid=self.neo4j_handler.run_uuid)
                
                nodes_data = []
                async for record in result_nodes:
                    nodes_data.append((record["id"], {"name": record["name"], "labels": record["labels"]}))
                G.add_nodes_from(nodes_data)
                
                # Fetch edges where both source and target are in the graph_uuid with type
                edge_query = """
                MATCH (n:GraphNode)-[r]->(m:GraphNode)
                WHERE n.graph_uuid = $graph_uuid AND m.graph_uuid = $graph_uuid
                RETURN elementId(n) as source, elementId(m) as target, type(r) as type
                """
                result_edges = await session.run(edge_query, graph_uuid=self.neo4j_handler.run_uuid)
                
                edges_data = []
                async for record in result_edges:
                    edges_data.append((record["source"], record["target"], {"type": record["type"]}))
                G.add_edges_from(edges_data)
                
            return G
        except Exception as e:
            logger.error(f"Error building NetworkX graph: {str(e)}")
            return nx.MultiDiGraph()

    def _are_same_context(self, relation_sentences: List[str], threshold: float = 0.8) -> Tuple[bool, int, List[int]]:
        """
        Determine representative relation and outliers using embedding similarity.
        Adapted from prune.py.
        """
        if len(relation_sentences) <= 1:
            return False, None, list(range(len(relation_sentences)))

        # 1. Encode embeddings
        embeddings = self.embedding_model.encode(relation_sentences)
        similarity_matrix = cosine_similarity(embeddings)

        # 2. Build graph based on similarity threshold
        G_sim = nx.Graph()
        G_sim.add_nodes_from(range(len(relation_sentences)))

        for i in range(len(relation_sentences)):
            for j in range(i + 1, len(relation_sentences)):
                if similarity_matrix[i][j] >= threshold:
                    G_sim.add_edge(i, j)

        # 3. Extract clusters (connected components)
        clusters = [list(c) for c in nx.connected_components(G_sim)]
        
        # If every sentence is isolated -> no pruning
        if all(len(c) == 1 for c in clusters):
            return False, None, list(range(len(relation_sentences)))

        # 4. Identify the main cluster (largest)
        clusters_sorted = sorted(clusters, key=len, reverse=True)
        main_cluster = clusters_sorted[0]

        # 5. Determine representative inside the main cluster
        main_embeddings = embeddings[main_cluster]
        centroid = np.mean(main_embeddings, axis=0)
        sims_to_centroid = cosine_similarity([centroid], main_embeddings)[0]
        rep_local_idx = np.argmax(sims_to_centroid)
        representative_idx = main_cluster[rep_local_idx]

        # 6. Remaining clusters = outliers
        outlier_indices = []
        for cluster in clusters_sorted[1:]:
            outlier_indices.extend(cluster)

        should_prune = len(main_cluster) > 1
        return should_prune, representative_idx, outlier_indices

    def _prune_redundant_edges(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Prune redundant edges between same nodes based on semantic similarity.
        Modifies the graph in-place or returns a copy.
        """
        logger.info("Starting semantic pruning of edges...")
        edges_to_remove = []
        
        # Identify unique node pairs that have edges
        # G.edges(keys=True) yields (u, v, key)
        # We need to group edges by (u, v)
        edge_groups = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            if (u, v) not in edge_groups:
                edge_groups[(u, v)] = []
            edge_groups[(u, v)].append((key, data))
            
        pruned_count = 0
        
        for (u, v), edges in edge_groups.items():
            if len(edges) <= 1:
                continue
                
            # Prepare sentences for embedding
            # Format: "NodeName RelationType NodeName"
            u_name = G.nodes[u].get('name', str(u))
            v_name = G.nodes[v].get('name', str(v))
            
            relation_sentences = []
            for _, data in edges:
                rel_type = data.get('type', '').replace("_", " ")
                sentence = f"{u_name} {rel_type} {v_name}"
                relation_sentences.append(sentence)
            
            should_prune, representative_idx, outlier_indices = self._are_same_context(relation_sentences)
            
            if should_prune:
                # Indices to keep: representative + outliers
                indices_to_keep = {representative_idx} | set(outlier_indices)
                
                # Identify keys to remove
                for idx, (key, _) in enumerate(edges):
                    if idx not in indices_to_keep:
                        edges_to_remove.append((u, v, key))
                        pruned_count += 1
                        
        if edges_to_remove:
            G.remove_edges_from(edges_to_remove)
            logger.info(f"Pruned {pruned_count} redundant edges.")
        else:
            logger.info("No redundant edges found to prune.")
            
        return G

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

    def get_llm_context(self) -> str:
        """
        Generate context string for LLM based on computed metrics.
        Provides structural insights, existing vocabulary, and recommendations.
        """
        if not self.metrics_cache:
            return "Graph context not yet available."
            
        if "error" in self.metrics_cache:
            return f"Graph context unavailable due to error: {self.metrics_cache['error']}"
            
        m = self.metrics_cache
        
        # Generate Recommendations
        recommendations = []
        if m.get("density", 0) > 0.1:
            recommendations.append("Graph is becoming dense. Focus on identifying distinct relationships to avoid redundancy.")
        elif m.get("density", 0) < 0.01:
            recommendations.append("Graph is sparse. Look for implicit connections between existing entities.")
            
        if m.get("weakly_connected_components", 0) > 1:
            recommendations.append(f"Graph has {m['weakly_connected_components']} disconnected components. Consider linking isolated clusters.")

        if m.get("hub_concentration", 0) > 0.4:
             recommendations.append("High hub concentration detected. Ensure central nodes are not becoming 'super-nodes' with irrelevant connections.")

        # Helper to safely get name string
        def get_safe_name(node_dict):
            name = node_dict.get('name')
            return str(name) if name else ""

        # Format Prompt
        prompt = f"""
        ## Current Knowledge Graph Context
        **Graph Size:** {m.get('node_count', 0)} nodes, {m.get('relationship_count', 0)} relationships
        **Density:** {m.get('density', 0)}
        **Connectivity:** {m.get('weakly_connected_components', 0)} components
        
        **Key Entities (Central Nodes):**
        {', '.join([get_safe_name(n) for n in m.get('top_central_nodes', [])]) if m.get('top_central_nodes') else 'None'}
        
        **Existing Relationship Types:**
        {', '.join([f"{r['type']} ({r['count']})" for r in m.get('top_relations', [])]) if m.get('top_relations') else 'None'}
        
        **Existing Entity Types:**
        {', '.join([f"{l['label']} ({l['count']})" for l in m.get('top_entity_types', [])]) if m.get('top_entity_types') else 'None'}
        
        **Recommendations:**
        {chr(10).join(f"- {rec}" for rec in recommendations) if recommendations else "- Maintain current graph quality."}
        
        **Instruction:** Check the list of Key Entities above. If a new entity is a synonym or refers to the same object, use the existing name.
        """
        return prompt.strip()