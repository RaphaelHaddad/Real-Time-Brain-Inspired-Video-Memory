from ..core.logger import get_logger
from ..core.config import CommunityHighGraphConfig
from ..components.neo4j_handler import Neo4jHandler
from typing import Dict, Any, List, Optional, Tuple
import time
import hashlib
import random
import statistics
import asyncio
import networkx as nx
import math

logger = get_logger(__name__)


class CommunityDetector:
    """
    Handles Incremental Leiden community detection for the community-based high-level graph.
    Designed for modularity and readability since this is a collaborative component.
    """
    
    def __init__(self, neo4j_handler: Neo4jHandler, config: CommunityHighGraphConfig, embedder=None):
        self.neo4j_handler = neo4j_handler
        self.config = config
        self.embedder = embedder
        self._last_community_state: Dict[str, str] = {}  # chunk_id -> community_id cache
        
    def compute_stable_community_id(self, member_chunk_ids: List[str]) -> str:
        """
        Compute a stable community ID from member chunk IDs.
        Using hash of sorted chunk IDs to ensure stability across runs.
        
        Args:
            member_chunk_ids: List of chunk IDs in the community
            
        Returns:
            16-character hex hash as stable community ID
        """
        sorted_ids = sorted(member_chunk_ids)
        hash_input = "-".join(sorted_ids).encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    async def detect_communities(self, G: nx.Graph) -> Dict[int, List[str]]:
        """
        Run Leiden community detection on the graph.
        
        Args:
            G: NetworkX graph (undirected)
            
        Returns:
            Dict mapping numeric community label -> list of node IDs
        """
        try:
            # Check if leiden is available (requires cdlib or leidenalg)
            try:
                import leidenalg
                import igraph as ig
                return await self._leiden_with_leidenalg(G)
            except ImportError:
                logger.warning("leidenalg not available, falling back to Louvain")
                return self._louvain_fallback(G)
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
    
    async def _leiden_with_leidenalg(self, G: nx.Graph) -> Dict[int, List[str]]:
        """Run Leiden using leidenalg library"""
        import leidenalg
        import igraph as ig
        
        # Convert NetworkX to igraph
        ig_graph = ig.Graph.from_networkx(G)
        
        # Run Leiden with configured resolution
        partition = leidenalg.find_partition(
            ig_graph, 
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.config.community_resolution
        )
        
        # Convert to dict format
        communities: Dict[int, List[str]] = {}
        for comm_idx, members in enumerate(partition):
            node_ids = [ig_graph.vs[m]['_nx_name'] for m in members]
            communities[comm_idx] = node_ids
        
        return communities
    
    def _louvain_fallback(self, G: nx.Graph) -> Dict[int, List[str]]:
        """Fallback to Louvain if Leiden not available"""
        if not hasattr(nx.community, 'louvain_communities'):
            logger.warning("Louvain not available in NetworkX version")
            return {}
        
        communities_list = nx.community.louvain_communities(
            G, 
            resolution=self.config.community_resolution,
            seed=42
        )
        
        communities: Dict[int, List[str]] = {}
        for comm_idx, members in enumerate(communities_list):
            communities[comm_idx] = list(members)
        
        return communities
    
    async def update_communities(self, embedder) -> Dict[str, Any]:
        """
        Main entry point: Run community detection and update the graph.
        
        Returns:
            Summary dict with stats about the community update
        """
        start_time = time.perf_counter()
        stats = {
            "num_communities": 0,
            "questions_per_community": {"avg": 0, "median": 0, "std": 0, "min": 0, "max": 0},
            "changed_communities": 0,
            "new_summaries_created": 0,
            "orphan_summaries_deleted": 0,
            "community_detection_time": 0,
            "embedding_time": 0,
            "total_time": 0
        }
        
        try:
            # Step 1: Build graph for community detection (Chunk nodes connected via Entity relationships)
            detection_start = time.perf_counter()
            G = await self._build_chunk_graph()
            
            if G.number_of_nodes() == 0:
                logger.info("No chunks found for community detection")
                return stats
            
            # Step 2: Run community detection
            communities = await self.detect_communities(G)
            stats["num_communities"] = len(communities)
            stats["community_detection_time"] = time.perf_counter() - detection_start
            
            if not communities:
                logger.warning("No communities detected")
                return stats
            
            # Step 3: Compute stable IDs and map chunks to communities
            chunk_to_community: Dict[str, str] = {}
            community_members: Dict[str, List[str]] = {}  # stable_id -> chunk_ids
            
            for _, member_ids in communities.items():
                # Filter to only Chunk node IDs
                chunk_ids = [nid for nid in member_ids if await self._is_chunk_node(nid)]
                if not chunk_ids:
                    continue
                    
                stable_id = self.compute_stable_community_id(chunk_ids)
                community_members[stable_id] = chunk_ids
                
                for chunk_id in chunk_ids:
                    chunk_to_community[chunk_id] = stable_id
            
            # Step 4: Identify changed communities
            changed = 0
            for chunk_id, new_comm in chunk_to_community.items():
                old_comm = self._last_community_state.get(chunk_id)
                if old_comm != new_comm:
                    changed += 1
            stats["changed_communities"] = changed
            
            # Step 5: Update chunk community_id properties
            await self.neo4j_handler.update_chunk_community_ids(chunk_to_community)
            self._last_community_state = chunk_to_community.copy()
            
            # Step 6: Compute questions per community statistics
            # Fetch all questions once per community (not per chunk!)
            questions_counts = []
            community_questions: Dict[str, List[str]] = {}  # stable_id -> all questions
            
            for stable_id, chunk_ids in community_members.items():
                # Call once per community, not per chunk
                chunks_data = await self.neo4j_handler.get_chunks_in_community(stable_id)
                all_questions = []
                for chunk in chunks_data:
                    all_questions.extend(chunk.get("questions", []))
                community_questions[stable_id] = list(set(all_questions))  # Deduplicate
                questions_counts.append(len(community_questions[stable_id]))
            
            if questions_counts:
                stats["questions_per_community"] = {
                    "avg": round(statistics.mean(questions_counts), 2),
                    "median": round(statistics.median(questions_counts), 2),
                    "std": round(statistics.stdev(questions_counts), 2) if len(questions_counts) > 1 else 0,
                    "min": min(questions_counts),
                    "max": max(questions_counts)
                }
                logger.info(f"📊 Questions per community: avg={stats['questions_per_community']['avg']}, "
                           f"median={stats['questions_per_community']['median']}, "
                           f"std={stats['questions_per_community']['std']}, "
                           f"min={stats['questions_per_community']['min']}, "
                           f"max={stats['questions_per_community']['max']}")
            
            # Step 7: Create/update CommunitySummary nodes with embeddings (parallel)
            embedding_start = time.perf_counter()
            summaries_created = await self._update_community_summaries(
                community_members, community_questions, embedder
            )
            stats["new_summaries_created"] = summaries_created
            stats["embedding_time"] = time.perf_counter() - embedding_start
            
            # Step 8: Delete orphan summaries
            valid_ids = list(community_members.keys())
            deleted = await self.neo4j_handler.delete_orphan_community_summaries(valid_ids)
            stats["orphan_summaries_deleted"] = deleted
            
            stats["total_time"] = time.perf_counter() - start_time
            
            # Logging
            logger.info(
                f"Community detection complete: {stats['num_communities']} communities, "
                f"questions_per_community avg={stats['questions_per_community']['avg']:.1f} "
                f"median={stats['questions_per_community']['median']:.1f} "
                f"std={stats['questions_per_community']['std']:.1f} "
                f"min={stats['questions_per_community']['min']} "
                f"max={stats['questions_per_community']['max']}, "
                f"detection_time={stats['community_detection_time']:.2f}s, "
                f"embedding_time={stats['embedding_time']:.2f}s"
            )
            
            # Debug: print random community descriptions
            if self.config.debug_print_community > 0 and community_questions:
                sample_size = min(self.config.debug_print_community, len(community_questions))
                sample_ids = random.sample(list(community_questions.keys()), sample_size)
                for comm_id in sample_ids:
                    questions = community_questions.get(comm_id, [])
                    logger.debug(f"Community {comm_id} questions ({len(questions)}): {questions[:5]}...")
            
            return stats
            
        except Exception as e:
            logger.error(f"Community update failed: {e}")
            stats["error"] = str(e)
            stats["total_time"] = time.perf_counter() - start_time
            return stats
    
    async def _build_chunk_graph(self) -> nx.Graph:
        """
        Build an undirected graph of Chunk nodes connected via shared entities.
        Two chunks are connected if they share at least one entity (via FROM_CHUNK relationships).
        """
        G = nx.Graph()
        
        try:
            async with self.neo4j_handler.driver.session() as session:
                # Get all chunks
                chunk_result = await session.run(
                    """
                    MATCH (c:Chunk:GraphNode {graph_uuid: $graph_uuid})
                    RETURN c.id as chunk_id
                    """,
                    graph_uuid=self.neo4j_handler.run_uuid
                )
                chunk_ids = [rec["chunk_id"] async for rec in chunk_result]
                G.add_nodes_from(chunk_ids)
                
                # Get chunk connections via shared entities
                edge_result = await session.run(
                    """
                    MATCH (c1:Chunk:GraphNode {graph_uuid: $graph_uuid})<-[:FROM_CHUNK]-(e:Entity)-[:FROM_CHUNK]->(c2:Chunk:GraphNode {graph_uuid: $graph_uuid})
                    WHERE c1.id < c2.id
                    RETURN DISTINCT c1.id as chunk1, c2.id as chunk2
                    """,
                    graph_uuid=self.neo4j_handler.run_uuid
                )
                edges = [(rec["chunk1"], rec["chunk2"]) async for rec in edge_result]
                G.add_edges_from(edges)
                
            return G
        except Exception as e:
            logger.error(f"Error building chunk graph: {e}")
            return nx.Graph()
    
    async def _is_chunk_node(self, node_id: str) -> bool:
        """Check if a node ID belongs to a Chunk node"""
        # Simple heuristic based on ID format
        # Chunk IDs typically contain underscores like "uuid_batch_chunk"
        return "_" in str(node_id)
    
    async def _update_community_summaries(
        self, 
        community_members: Dict[str, List[str]], 
        community_questions: Dict[str, List[str]],
        embedder
    ) -> int:
        """
        Create/update CommunitySummary nodes with chunked embedding computation.
        
        Uses sparse-to-dense approach: splits large question sets into overlapping chunks,
        embeds each chunk separately, stores all chunk embeddings for MaxSim retrieval.
        
        Returns:
            Number of summaries created/updated
        """
        if not embedder:
            logger.warning("No embedder available for community summaries")
            return 0
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Configuration for chunking
        MAX_TOKENS = 450  # Safe limit below 512 for most embedding models
        CHUNK_OVERLAP = 100  # Token overlap between chunks
        
        # Estimate ~4 characters per token
        max_chars = MAX_TOKENS * 4
        overlap_chars = CHUNK_OVERLAP * 4
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", "? ", " ", ""],
            length_function=len
        )
        
        created = 0
        
        for stable_id, questions in community_questions.items():
            if not questions:
                continue
                
            # Concatenate questions with numbering for context
            content = " ".join([f"Q{i+1}. {q}" for i, q in enumerate(questions)])
            
            # Split into chunks if too long
            content_chunks = splitter.split_text(content) if len(content) > max_chars else [content]
            
            # Embed all chunks in parallel
            try:
                embedding_tasks = [embedder.aembed_query(chunk) for chunk in content_chunks]
                chunk_embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
                
                # Filter out failed embeddings
                valid_embeddings = []
                valid_chunks = []
                for chunk, emb in zip(content_chunks, chunk_embeddings):
                    if isinstance(emb, Exception):
                        logger.warning(f"Embedding failed for community {stable_id} chunk: {emb}")
                        continue
                    valid_embeddings.append(emb)
                    valid_chunks.append(chunk)
                
                if not valid_embeddings:
                    logger.warning(f"No valid embeddings for community {stable_id}, skipping")
                    continue
                
                member_ids = community_members.get(stable_id, [])
                
                # Store with chunked embeddings
                success = await self.neo4j_handler.upsert_community_summary(
                    community_id=stable_id,
                    content=content,  # Full content for reference
                    embedding=valid_embeddings[0],  # Primary embedding (first chunk)
                    member_chunk_ids=member_ids,
                    question_chunks=valid_chunks,
                    embedding_chunks=valid_embeddings
                )
                if success:
                    created += 1
                    if len(valid_chunks) > 1:
                        logger.debug(f"Community {stable_id}: split into {len(valid_chunks)} embedding chunks")
                        
            except Exception as e:
                logger.error(f"Error processing community {stable_id}: {e}")
                continue
        
        return created


class ACSAutomata:
    """
    ACS (Adaptive Computing System) Automata
    Computes network science metrics using NetworkX (in-memory) for reliability and simplicity.
    Optionally runs community detection when community_high_graph is enabled.
    """
    
    def __init__(self, neo4j_handler: Neo4jHandler, community_config: CommunityHighGraphConfig = None, embedder = None):
        self.neo4j_handler = neo4j_handler
        self.metrics_cache = {}
        self.last_update_time = time.time()
        self.community_config = community_config
        self.embedder = embedder
        
        # Initialize community detector if enabled
        self.community_detector: Optional[CommunityDetector] = None
        if community_config and community_config.community_creator:
            self.community_detector = CommunityDetector(neo4j_handler, community_config, embedder)
            logger.info("Community detection enabled in ACS Automata")
    
    def set_embedder(self, embedder):
        """Set or update the embedder used for community summaries"""
        self.embedder = embedder
        if self.community_detector:
            self.community_detector.embedder = embedder
    
    async def run_community_detection_if_due(self, batch_idx: int) -> Optional[Dict[str, Any]]:
        """
        Run community detection if the batch index indicates it's due.
        
        Args:
            batch_idx: Current batch index (0-indexed)
            
        Returns:
            Community stats dict if run, None otherwise
        """
        if not self.community_detector or not self.community_config:
            return None
        
        frequency = self.community_config.frequency_incremental_leiden
        # Run on batches: frequency-1, 2*frequency-1, 3*frequency-1, etc. (every N batches)
        # batch_idx is 0-indexed, so we check (batch_idx + 1) % frequency == 0
        if (batch_idx + 1) % frequency == 0:
            logger.info(f"Running scheduled community detection at batch {batch_idx + 1}")
            return await self.community_detector.update_communities(self.embedder)
        
        return None
        
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