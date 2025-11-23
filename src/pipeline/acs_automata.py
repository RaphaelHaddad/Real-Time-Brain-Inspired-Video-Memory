from ..core.logger import get_logger
from ..components.neo4j_handler import Neo4jHandler
from typing import Dict, Any
import time
import networkx as nx

logger = get_logger(__name__)

class ACSAutomata:
    """
    ACS (Adaptive Computing System) Automata
    Computes network science metrics that can be updated efficiently after each batch
    """
    
    def __init__(self, neo4j_handler: Neo4jHandler):
        self.neo4j_handler = neo4j_handler
        self.metrics_cache = {}
        self.last_update_time = time.time()
        
    async def update_metrics(self) -> Dict[str, Any]:
        """Update and compute network science metrics"""
        start_time = time.perf_counter()
        
        try:
            logger.info("Computing network science metrics...")
            
            # Compute basic graph metrics
            node_count = await self._compute_node_count()
            rel_count = await self._compute_relationship_count()
            metrics = {
                "node_count": node_count,
                "relationship_count": rel_count,
                "density": await self._compute_density(),
                "avg_degree": await self._compute_avg_degree(),
                "diameter_estimate": await self._compute_diameter_estimate(),
                "clustering_coefficient": await self._compute_clustering_coefficient(),
                "computational_time": time.perf_counter() - start_time
            }

            # Additional network-science metrics using GDS where available (best-effort)
            try:
                # Weakly connected components (WCC) count and largest component size
                wcc_query = """
                CALL gds.wcc.stream({nodeProjection: '__ALL__', relationshipProjection: '__ALL__'})
                YIELD nodeId, componentId
                RETURN componentId, count(*) as size
                ORDER BY size DESC
                """
                async with self.neo4j_handler.driver.session() as session:
                    res = await session.run(wcc_query)
                    comps = [rec async for rec in res]
                    wcc_count = len(comps)
                    largest_wcc = comps[0]['size'] if comps else 0
            except Exception as e:
                logger.error(f"Failed to compute WCC metrics: {str(e)}")
                wcc_count = None
                largest_wcc = None

            metrics.update({
                "weakly_connected_components": wcc_count,
                "largest_wcc_size": largest_wcc
            })

            # PageRank top-10 percent
            try:
                pr_query = """
                CALL gds.pageRank.stream({nodeProjection: '__ALL__', relationshipProjection: '__ALL__'})
                YIELD nodeId, score
                RETURN score
                ORDER BY score DESC
                """
                async with self.neo4j_handler.driver.session() as session:
                    res = await session.run(pr_query)
                    scores = [float(rec['score']) async for rec in res]
                    total_pr = sum(scores) if scores else 0.0
                    top10 = scores[:10]
                    pr_top10_pct = (sum(top10) / total_pr * 100.0) if total_pr > 0 else None
            except Exception as e:
                logger.error(f"Failed to compute PageRank top10 percent: {str(e)}")
                pr_top10_pct = None

            metrics.update({
                "pagerank_top10_percent": pr_top10_pct
            })

            # Louvain communities count and (best-effort) modularity
            try:
                louv_query = """
                CALL gds.louvain.stream({nodeProjection: '__ALL__', relationshipProjection: '__ALL__'})
                YIELD nodeId, communityId
                RETURN communityId, count(*) as size
                ORDER BY size DESC
                """
                async with self.neo4j_handler.driver.session() as session:
                    res = await session.run(louv_query)
                    comms = [rec async for rec in res]
                    community_count = len(comms)
                    # modularity not directly provided by stream; set None if not available
                    modularity = None
            except Exception as e:
                logger.error(f"Failed to compute Louvain communities: {str(e)}")
                community_count = None
                modularity = None

            metrics.update({
                "louvain_communities": community_count,
                "louvain_modularity": modularity
            })

            # Label entropy (Shannon) based on node labels distribution
            try:
                async with self.neo4j_handler.driver.session() as session:
                    lbl_q = """
                    MATCH (n:GraphNode) WHERE n.graph_uuid = $graph_uuid
                    UNWIND labels(n) as lab
                    RETURN lab, count(*) as cnt
                    """
                    res = await session.run(lbl_q, graph_uuid=self.neo4j_handler.run_uuid)
                    label_counts = {rec['lab']: rec['cnt'] async for rec in res}
                    # compute Shannon entropy
                    import math
                    total = sum(label_counts.values())
                    entropy = 0.0
                    if total > 0:
                        for c in label_counts.values():
                            p = c / total
                            entropy -= p * math.log(p, 2)
                    label_entropy = round(entropy, 4)
            except Exception as e:
                logger.error(f"Failed to compute label entropy: {str(e)}")
                label_entropy = None

            metrics.update({
                "label_entropy": label_entropy
            })
            
            # Check if we need fallback for any null GDS metrics
            if (metrics.get("weakly_connected_components") is None or 
                metrics.get("pagerank_top10_percent") is None or 
                metrics.get("louvain_communities") is None):
                
                fallback_metrics = await self._compute_networkx_fallback()
                # Only update keys that are None
                for k, v in fallback_metrics.items():
                    if metrics.get(k) is None:
                        metrics[k] = v

            # Log any null metrics for diagnostics
            null_metrics = [k for k, v in metrics.items() if v is None]
            if null_metrics:
                logger.error(f"Null metrics detected for run_uuid {self.neo4j_handler.run_uuid}: {null_metrics}")
            
            self.metrics_cache.update(metrics)
            self.last_update_time = time.time()
            
            logger.info(f"Network metrics computed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing network metrics: {str(e)}")
            return {"error": str(e), "computational_time": time.perf_counter() - start_time}

    async def _compute_node_count(self) -> int:
        """Compute the number of nodes in the graph"""
        return await self.neo4j_handler.get_node_count()

    async def _compute_relationship_count(self) -> int:
        """Compute the number of relationships in the graph"""
        return await self.neo4j_handler.get_relationship_count()

    async def _compute_density(self) -> float:
        """Compute graph density (ratio of actual edges to possible edges)"""
        try:
            node_count = await self._compute_node_count()
            rel_count = await self._compute_relationship_count()
            
            if node_count < 2:
                return 0.0
                
            # For directed graphs, max possible edges is n*(n-1)
            max_possible_edges = node_count * (node_count - 1)
            density = rel_count / max_possible_edges if max_possible_edges > 0 else 0.0
            
            return round(density, 4)
        except Exception as e:
            logger.error(f"Error computing density: {str(e)}")
            return 0.0

    async def _compute_avg_degree(self) -> float:
        """Compute average node degree"""
        try:
            node_count = await self._compute_node_count()
            rel_count = await self._compute_relationship_count()
            
            if node_count == 0:
                return 0.0
            
            # In a directed graph, avg degree is total relationships / nodes
            avg_degree = (2 * rel_count) / node_count if node_count > 0 else 0.0
            return round(avg_degree, 4)
        except Exception as e:
            logger.error(f"Error computing avg degree: {str(e)}")
            return 0.0

    async def _compute_diameter_estimate(self) -> int:
        """Estimate the diameter of the graph (will implement a simple estimation)"""
        # For now, we'll return a placeholder value
        # In a real implementation, this would be more complex
        try:
            node_count = await self._compute_node_count()
            if node_count == 0:
                return 0
            elif node_count < 10:
                return min(node_count, 5)  # Small graph
            else:
                return min(node_count // 2, 50)  # Estimate for larger graphs
        except Exception as e:
            logger.error(f"Error computing diameter estimate: {str(e)}")
            return 0

    async def _compute_clustering_coefficient(self) -> float:
        """Compute the clustering coefficient (simplified estimation)"""
        try:
            node_count = await self._compute_node_count()
            rel_count = await self._compute_relationship_count()
            
            if node_count < 2:
                return 0.0
            
            # Simplified estimation of clustering coefficient
            # In real implementation, this would require actual triangle counting
            density = await self._compute_density()
            clustering = min(density * 2, 1.0)  # Rough estimation
            
            return round(clustering, 4)
        except Exception as e:
            logger.error(f"Error computing clustering coefficient: {str(e)}")
            return 0.0

    async def _compute_networkx_fallback(self) -> Dict[str, Any]:
        """Compute metrics using NetworkX when GDS is unavailable"""
        try:
            logger.info("GDS unavailable, falling back to NetworkX for metrics...")
            G = nx.DiGraph()
            async with self.neo4j_handler.driver.session() as session:
                # Fetch nodes
                result_nodes = await session.run("MATCH (n) RETURN elementId(n) as id")
                nodes = [record["id"] async for record in result_nodes]
                G.add_nodes_from(nodes)
                
                # Fetch edges
                result_edges = await session.run("MATCH (n)-[r]->(m) RETURN elementId(n) as source, elementId(m) as target")
                edges = [(record["source"], record["target"]) async for record in result_edges]
                G.add_edges_from(edges)
            
            if len(G) == 0:
                return {}

            metrics = {}
            
            # WCC
            wcc = list(nx.weakly_connected_components(G))
            metrics["weakly_connected_components"] = len(wcc)
            metrics["largest_wcc_size"] = max(len(c) for c in wcc) if wcc else 0
            
            # PageRank
            try:
                pagerank = nx.pagerank(G)
                scores = list(pagerank.values())
                total_pr = sum(scores)
                # Top 10 nodes share
                scores.sort(reverse=True)
                top10 = scores[:10]
                metrics["pagerank_top10_percent"] = (sum(top10) / total_pr * 100.0) if total_pr > 0 else 0
            except Exception:
                metrics["pagerank_top10_percent"] = None

            # Louvain (requires undirected)
            try:
                G_undir = G.to_undirected()
                # Check if louvain is available
                if hasattr(nx.community, 'louvain_communities'):
                    louvain_comms = nx.community.louvain_communities(G_undir)
                    metrics["louvain_communities"] = len(louvain_comms)
                    metrics["louvain_modularity"] = nx.community.modularity(G_undir, louvain_comms)
                else:
                    metrics["louvain_communities"] = None
                    metrics["louvain_modularity"] = None
            except Exception:
                metrics["louvain_communities"] = None
                metrics["louvain_modularity"] = None
                
            return metrics
        except Exception as e:
            logger.error(f"NetworkX fallback failed: {e}")
            return {}