import networkx as nx
from typing import Dict, Any, List, Tuple
import numpy as np
from collections import Counter
from ..core.logger import get_logger

logger = get_logger(__name__)

class NetworkMetrics:
    """
    Provides network science measures to guide graph construction.
    These metrics help LLMs understand the current graph structure and make informed decisions.
    """
    
    def __init__(self, neo4j_handler):
        self.neo4j = neo4j_handler
    
    async def compute_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive network science metrics for guiding LLM.
        
        Returns metrics across multiple dimensions:
        - Structural: density, clustering, diameter
        - Centrality: key nodes and their importance
        - Community: modularity, community structure
        - Growth: preferential attachment indicators
        - Semantic: relation diversity, entity types
        """
        
        # Fetch current graph state
        graph_data = await self._fetch_graph_data()
        
        if not graph_data['nodes']:
            return self._empty_graph_metrics()
        
        # Build NetworkX graph for analysis
        G = self._build_networkx_graph(graph_data)
        
        metrics = {
            "basic_stats": self._compute_basic_stats(G, graph_data),
            "structural_metrics": self._compute_structural_metrics(G),
            "centrality_metrics": self._compute_centrality_metrics(G),
            "community_metrics": self._compute_community_metrics(G),
            "semantic_metrics": self._compute_semantic_metrics(graph_data),
            "growth_indicators": self._compute_growth_indicators(G),
            "llm_guidance": self._generate_llm_guidance(G, graph_data)
        }
        
        return metrics
    
    async def _fetch_graph_data(self) -> Dict[str, Any]:
        """Fetch current graph structure from Neo4j"""
        query = """
        MATCH (n)
        WHERE n.run_uuid = $run_uuid
        WITH n, COUNT { (n)--() } as degree
        OPTIONAL MATCH (n)-[r]->(m)
        WHERE m.run_uuid = $run_uuid
        WITH collect(DISTINCT {
            id: id(n),
            name: n.name,
            labels: labels(n),
            degree: degree
        }) as nodes,
        collect({
            from_id: id(n),
            to_id: id(m),
            type: type(r),
            from_name: n.name,
            to_name: m.name
        }) as relationships
        RETURN nodes, relationships
        """
        
        try:
            # Use the Neo4j handler's session to execute the query
            async with self.neo4j.driver.session() as session:
                result = await session.run(query, {"run_uuid": self.neo4j.run_uuid})
                record = await result.single()
                
                if record:
                    return {
                        'nodes': record['nodes'],
                        'relationships': [r for r in record['relationships'] if r['to_id'] is not None]
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch graph data: {e}")
        
        return {'nodes': [], 'relationships': []}
    
    def _build_networkx_graph(self, graph_data: Dict) -> nx.Graph:
        """Convert Neo4j data to NetworkX graph"""
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'], name=node['name'], labels=node.get('labels', []))
        
        # Add edges
        for rel in graph_data['relationships']:
            G.add_edge(rel['from_id'], rel['to_id'], type=rel['type'])
        
        return G
    
    def _compute_basic_stats(self, G: nx.Graph, graph_data: Dict) -> Dict[str, Any]:
        """Basic graph statistics"""
        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            "num_components": nx.number_connected_components(G)
        }
    
    def _compute_structural_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Structural properties: clustering, diameter, assortativity"""
        if G.number_of_nodes() < 2:
            return {"clustering_coefficient": 0.0, "diameter": 0, "assortativity": 0.0, "transitivity": 0.0, "average_shortest_path": 0}
        
        try:
            diameter = nx.diameter(G) if nx.is_connected(G) else max(
                nx.diameter(G.subgraph(c)) for c in nx.connected_components(G)
            )
        except:
            diameter = 0
        
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except:
            assortativity = 0.0
        
        return {
            "clustering_coefficient": nx.average_clustering(G),
            "transitivity": nx.transitivity(G),
            "diameter": diameter,
            "average_shortest_path": nx.average_shortest_path_length(G) if nx.is_connected(G) else 0,
            "assortativity": assortativity
        }
    
    def _compute_centrality_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Identify key nodes using multiple centrality measures"""
        if G.number_of_nodes() == 0:
            return {"top_nodes": [], "prominent_entities": [], "centralization": {"degree": 0, "betweenness": 0}}
        
        # Compute various centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        
        # PageRank for importance
        try:
            pagerank = nx.pagerank(G)
        except:
            pagerank = {node: 0 for node in G.nodes()}
        
        # Aggregate scores
        node_scores = {}
        for node in G.nodes():
            node_scores[node] = {
                "name": G.nodes[node].get('name', str(node)),
                "degree": degree_cent.get(node, 0),
                "betweenness": betweenness_cent.get(node, 0),
                "closeness": closeness_cent.get(node, 0),
                "pagerank": pagerank.get(node, 0),
                "combined_score": (
                    degree_cent.get(node, 0) * 0.3 +
                    betweenness_cent.get(node, 0) * 0.3 +
                    closeness_cent.get(node, 0) * 0.2 +
                    pagerank.get(node, 0) * 0.2
                )
            }
        
        # Sort nodes by score
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)

        # Top 10 most important nodes for detailed stats
        top_nodes = sorted_nodes[:10]
        
        # Top 50 entity names for context injection (to avoid duplicates)
        prominent_entities = [score['name'] for _, score in sorted_nodes[:50]]
        
        return {
            "top_nodes": [
                {
                    "name": score['name'],
                    "degree": round(score['degree'], 3),
                    "betweenness": round(score['betweenness'], 3),
                    "pagerank": round(score['pagerank'], 3),
                    "combined_score": round(score['combined_score'], 3)
                }
                for _, score in top_nodes
            ],
            "prominent_entities": prominent_entities,
            "centralization": {
                "degree": max(degree_cent.values()) - sum(degree_cent.values()) / len(degree_cent) if degree_cent else 0,
                "betweenness": max(betweenness_cent.values()) if betweenness_cent else 0
            }
        }
    
    def _compute_community_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Community detection and modularity"""
        if G.number_of_nodes() < 2:
            return {"num_communities": 0, "modularity": 0.0}
        
        try:
            # Louvain community detection
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G)
            modularity = community.modularity(G, communities)
            
            # Community sizes
            community_sizes = [len(c) for c in communities]
            
            return {
                "num_communities": len(communities),
                "modularity": round(modularity, 3),
                "largest_community_size": max(community_sizes) if community_sizes else 0,
                "smallest_community_size": min(community_sizes) if community_sizes else 0,
                "average_community_size": round(np.mean(community_sizes), 2) if community_sizes else 0
            }
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {"num_communities": 0, "modularity": 0.0}
    
    def _compute_semantic_metrics(self, graph_data: Dict) -> Dict[str, Any]:
        """Semantic diversity and relation patterns"""
        # Relation type distribution
        relation_types = [rel['type'] for rel in graph_data['relationships']]
        relation_counts = Counter(relation_types)
        
        # Entity label distribution
        entity_labels = []
        for node in graph_data['nodes']:
            entity_labels.extend(node.get('labels', []))
        label_counts = Counter(entity_labels)
        
        return {
            "unique_relation_types": len(relation_counts),
            "top_relations": [
                {"type": rel, "count": count}
                for rel, count in relation_counts.most_common(20)  # Increased from 10 to 20
            ],
            "relation_diversity": round(len(relation_counts) / len(relation_types), 3) if relation_types else 0,
            "unique_entity_labels": len(label_counts),
            "top_entity_types": [
                {"label": label, "count": count}
                for label, count in label_counts.most_common(5)
            ]
        }
    
    def _compute_growth_indicators(self, G: nx.Graph) -> Dict[str, Any]:
        """Indicators for preferential attachment and growth patterns"""
        if G.number_of_edges() == 0:
            return {"rich_get_richer": False, "hub_concentration": 0.0, "max_degree": 0, "average_degree": 0, "degree_variance": 0, "degree_distribution": {}}
        
        degrees = [d for _, d in G.degree()]
        degree_dist = Counter(degrees)
        
        # Check for scale-free properties (power law)
        max_degree = max(degrees) if degrees else 0
        avg_degree = np.mean(degrees) if degrees else 0
        
        # Hub concentration (top 10% nodes)
        top_10_percent = max(1, int(0.1 * G.number_of_nodes()))
        if top_10_percent > 0:
            top_degrees = sorted(degrees, reverse=True)[:top_10_percent]
            hub_concentration = sum(top_degrees) / sum(degrees) if sum(degrees) > 0 else 0
        else:
            hub_concentration = 0
        
        return {
            "max_degree": max_degree,
            "average_degree": round(avg_degree, 2),
            "degree_variance": round(np.var(degrees), 2) if degrees else 0,
            "hub_concentration": round(hub_concentration, 3),
            "rich_get_richer": hub_concentration > 0.3,  # Heuristic threshold
            "degree_distribution": dict(sorted(degree_dist.items())[:10])
        }
    
    def _generate_llm_guidance(self, G: nx.Graph, graph_data: Dict) -> Dict[str, Any]:
        """Generate actionable guidance for LLM"""
        metrics = self._compute_basic_stats(G, graph_data)
        centrality = self._compute_centrality_metrics(G)
        semantic = self._compute_semantic_metrics(graph_data)
        
        # Generate recommendations
        recommendations = []
        
        if metrics['density'] > 0.7:
            recommendations.append("Graph is highly dense. Focus on identifying distinct relationships to avoid redundancy.")
        elif metrics['density'] < 0.1:
            recommendations.append("Graph is sparse. Look for implicit connections between existing entities.")
        
        if metrics['num_components'] > 1:
            recommendations.append(f"Graph has {metrics['num_components']} disconnected components. Consider linking isolated clusters.")
        
        if semantic['relation_diversity'] < 0.3:
            recommendations.append("Low relation diversity. Explore more varied relationship types.")
        
        # Extract key entity names for context
        key_entities = [node['name'] for node in centrality['top_nodes'][:5]]
        
        return {
            "recommendations": recommendations,
            "key_entities_in_graph": key_entities,
            "graph_maturity": self._assess_maturity(metrics, semantic),
            "focus_areas": self._suggest_focus_areas(metrics, semantic)
        }
    
    def _assess_maturity(self, metrics: Dict, semantic: Dict) -> str:
        """Assess graph maturity level"""
        if metrics['node_count'] < 10:
            return "nascent"
        elif metrics['node_count'] < 50:
            return "developing"
        elif metrics['node_count'] < 200:
            return "mature"
        else:
            return "extensive"
    
    def _suggest_focus_areas(self, metrics: Dict, semantic: Dict) -> List[str]:
        """Suggest what to focus on next"""
        areas = []
        
        if semantic['unique_relation_types'] < 5:
            areas.append("diversify_relations")
        
        if metrics['num_components'] > 1:
            areas.append("connect_components")
        
        if metrics['average_degree'] < 2:
            areas.append("increase_connectivity")
        
        return areas if areas else ["maintain_quality"]
    
    def _empty_graph_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for new graph"""
        return {
            "basic_stats": {"node_count": 0, "edge_count": 0, "density": 0.0, "average_degree": 0, "is_connected": False, "num_components": 0},
            "structural_metrics": {"clustering_coefficient": 0.0, "diameter": 0, "assortativity": 0.0, "transitivity": 0.0, "average_shortest_path": 0},
            "centrality_metrics": {"top_nodes": [], "centralization": {"degree": 0, "betweenness": 0}},
            "community_metrics": {"num_communities": 0, "modularity": 0.0},
            "semantic_metrics": {"unique_relation_types": 0, "top_relations": [], "relation_diversity": 0, "unique_entity_labels": 0, "top_entity_types": []},
            "growth_indicators": {"rich_get_richer": False, "hub_concentration": 0.0, "max_degree": 0, "average_degree": 0, "degree_variance": 0, "degree_distribution": {}},
            "llm_guidance": {
                "recommendations": ["This is a new graph. Start by identifying core entities and their relationships."],
                "key_entities_in_graph": [],
                "graph_maturity": "empty",
                "focus_areas": ["establish_foundation"]
            }
        }
    
    def format_for_llm_prompt(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into human-readable prompt for LLM"""
        basic = metrics['basic_stats']
        guidance = metrics['llm_guidance']
        
        prompt = f"""
        ## Current Knowledge Graph Context

        **Graph Size:** {basic['node_count']} nodes, {basic['edge_count']} relationships
        **Maturity Level:** {guidance['graph_maturity']}
        **Density:** {basic.get('density', 0):.3f} (0=sparse, 1=complete)
        **Connectivity:** {'Connected' if basic.get('is_connected') else f"{basic.get('num_components', 0)} separate components"}

        **Key Entities Already Present:**
        {', '.join(guidance['key_entities_in_graph'][:10]) if guidance['key_entities_in_graph'] else 'None yet'}

        **Recommendations for This Batch:**
        {chr(10).join(f"- {rec}" for rec in guidance['recommendations'])}

        **Focus Areas:** {', '.join(guidance['focus_areas'])}
        """
        
        # Add centrality info if available
        if 'centrality_metrics' in metrics and metrics['centrality_metrics']['top_nodes']:
            prompt += "\n**Most Important Nodes (by centrality):**\n"
            for node in metrics['centrality_metrics']['top_nodes'][:5]:
                prompt += f"- {node['name']} (PageRank: {node['pagerank']:.3f})\n"
        
        # Add Schema/Vocabulary section for consistency
        prompt += "\n## Existing Graph Vocabulary (Reuse if necessary to avoid repitition)\n"
        
        # Relations
        if 'semantic_metrics' in metrics:
            sem = metrics['semantic_metrics']
            rels = [r['type'] for r in sem.get('top_relations', [])]
            if rels:
                prompt += f"**Existing Relationship Types:** {', '.join(rels)}\n"

        # Entities (Expanded list)
        if 'centrality_metrics' in metrics:
            cent = metrics['centrality_metrics']
            entities = cent.get('prominent_entities', [])
            # Filter out empty names and duplicates
            entities = sorted(list(set([e for e in entities if e])))
            if entities:
                prompt += f"**Common Entities:** {', '.join(entities)}\n"
            
        prompt += "\n**Instruction:** Check the list of Common Entities above. If a new entity is a synonym or refers to the same object, use the existing name."
        
        return prompt.strip()