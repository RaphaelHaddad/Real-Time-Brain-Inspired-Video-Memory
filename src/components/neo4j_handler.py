from neo4j import AsyncGraphDatabase, AsyncManagedTransaction
from typing import List, Dict, Any, Optional
import time
import json
from ..core.logger import get_logger
from langchain_openai import OpenAIEmbeddings
import uuid
import logging
import asyncio

logger = get_logger(__name__)

class Neo4jHandler:
    def __init__(self, neo4j_config, kg_config, run_uuid: str):
        self.neo4j_config = neo4j_config
        self.kg_config = kg_config
        self.run_uuid = run_uuid
        self.driver = AsyncGraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password),
            database=neo4j_config.database
        )
        # Suppress Neo4j warnings if not verbose
        if not kg_config.verbose:
            logging.getLogger('neo4j').setLevel(logging.ERROR)
        # Initialize embedding client
        self.embedder = OpenAIEmbeddings(
            model=kg_config.embedding_model,
            openai_api_key=kg_config.embedding_api_key,
            openai_api_base=kg_config.embedding_endpoint
        )

        logger.info(f"Initialized Neo4j handler for run UUID: {run_uuid}")

    async def verify_connection(self) -> bool:
        """Verify that Neo4j is accessible by running a simple query"""
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 'Neo4j is accessible' as message")
                record = await result.single()
                if record and record["message"] == "Neo4j is accessible":
                    logger.info("Neo4j connection verified successfully")
                    return True
                else:
                    logger.error("Neo4j connection verification failed: unexpected response")
                    return False
        except Exception as e:
            logger.error(f"Neo4j connection verification failed: {str(e)}")
            return False

    async def _initialize_database(self):
        """Initialize database schema and constraints"""
        async with self.driver.session() as session:
            # Create constraints for UUID-based isolation
            await session.execute_write(self._create_constraints)
            await session.execute_write(self._create_indexes)

    async def _create_constraints(self, tx):
        """Create database constraints"""
        # We'll create constraints to ensure graph isolation by run_uuid
        await tx.run("CREATE CONSTRAINT graph_uuid IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.graph_uuid IS UNIQUE")
        logger.info("Created graph UUID constraint")

    async def _create_indexes(self, tx):
        """Create database indexes"""
        # Create regular indexes for faster queries
        await tx.run("CREATE INDEX graph_uuid_index IF NOT EXISTS FOR (n:GraphNode) ON (n.graph_uuid)")
        await tx.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")
        await tx.run("CREATE INDEX chunk_time_index IF NOT EXISTS FOR (n:Chunk) ON (n.time)")
        
        # Create fulltext index for entity search (required for retrieval)
        try:
            await tx.run("""
                CREATE FULLTEXT INDEX entityName IF NOT EXISTS
                FOR (n:Entity)
                ON EACH [n.name]
            """)
            logger.info("Created fulltext index for entity search")
        except Exception as e:
            logger.warning(f"Fulltext index creation skipped (may already exist): {e}")
        
        logger.info("Created graph indexes")

    async def add_batch_to_graph(self, triplets: List[Dict[str, Any]], batch_data: List[Dict], batch_idx: int = 0, text_chunks: List[Dict[str, Any]] = None, operations: Dict[str, Any] = None) -> Dict[str, float]:
        """Add a batch of triplets AND text chunks to the graph (hybrid mode)"""
        start_time = time.perf_counter()
        timings = {
            "embedding_time": 0,
            "indexing_time": 0,
            "graph_injection_time": 0
        }
        
        try:
            async with self.driver.session() as session:
                logger.info(f"Adding {len(triplets)} triplets + {len(batch_data)} chunks to graph (hybrid mode)")
                
                # Single unified injection: triplets + chunks
                injection_start = time.perf_counter()
                batch_time = batch_data[-1].get('time', '') if batch_data else ""
                
                # Create triplets (with source_chunks tracking)
                await self._create_triplets(session, triplets, batch_time, batch_idx)
                
                # Always create chunk nodes for hybrid retrieval
                await self._create_chunks_with_embeddings(session, batch_data, triplets, batch_idx, text_chunks)
                
                # After creating triplets and chunks, optionally apply operations (merge/link/prune)
                if operations:
                    # Log counts before operations
                    try:
                        nodes_before = await session.run(
                            "MATCH (n:GraphNode) WHERE n.graph_uuid = $graph_uuid RETURN count(n) as cnt",
                            graph_uuid=self.run_uuid
                        )
                        rels_before = await session.run(
                            "MATCH ()-[r]->() WHERE r.graph_uuid = $graph_uuid RETURN count(r) as cnt",
                            graph_uuid=self.run_uuid
                        )
                        nb = (await nodes_before.single())['cnt'] if nodes_before else 0
                        rb = (await rels_before.single())['cnt'] if rels_before else 0
                        logger.debug(f"Graph counts before operations: nodes={nb}, relationships={rb}")
                    except Exception as e:
                        logger.debug(f"Failed to fetch counts before operations: {e}")

                    try:
                        await self._apply_operations(session, operations, batch_idx)
                    except Exception as e:
                        logger.warning(f"Failed to apply operations: {e}")

                    # Log counts after operations
                    try:
                        nodes_after = await session.run(
                            "MATCH (n:GraphNode) WHERE n.graph_uuid = $graph_uuid RETURN count(n) as cnt",
                            graph_uuid=self.run_uuid
                        )
                        rels_after = await session.run(
                            "MATCH ()-[r]->() WHERE r.graph_uuid = $graph_uuid RETURN count(r) as cnt",
                            graph_uuid=self.run_uuid
                        )
                        na = (await nodes_after.single())['cnt'] if nodes_after else 0
                        ra = (await rels_after.single())['cnt'] if rels_after else 0
                        logger.debug(f"Graph counts after operations: nodes={na}, relationships={ra}")
                    except Exception as e:
                        logger.debug(f"Failed to fetch counts after operations: {e}")
                
                injection_time = time.perf_counter() - injection_start
                timings["graph_injection_time"] = injection_time
                
                # Ensure indexes exist
                indexing_start = time.perf_counter()
                await self._ensure_indexes(session)
                indexing_time = time.perf_counter() - indexing_start
                timings["indexing_time"] = indexing_time
                
                total_time = time.perf_counter() - start_time
                logger.info(f"Batch injected in {total_time:.2f}s (triplets: {injection_time:.2f}s, indexes: {indexing_time:.2f}s)")
                
                return timings
                
        except Exception as e:
            logger.error(f"Error adding batch to graph: {str(e)}")
            raise

    async def _create_triplets(self, session, triplets: List[Dict[str, Any]], batch_time: str = "", batch_idx: int = 0):
        """Create triplets in Neo4j, preserving source_chunks metadata"""
        if not triplets:
            return
            
        # Batch create entities and relationships efficiently
        for triplet in triplets:
            head = triplet.get('head', '').strip()
            relation = triplet.get('relation', '').strip()
            tail = triplet.get('tail', '').strip()
            source_chunks = triplet.get('source_chunks', [])
            
            if not head or not relation or not tail:
                continue
                
            # Create head entity
            await session.run(
                """
                MERGE (h:Entity:GraphNode {name: $head, graph_uuid: $graph_uuid})
                SET h.created_at = datetime(),
                    h.batch_time = $batch_time,
                    h.batch_id = $batch_idx,
                    h.source_chunks = $source_chunks
                """,
                head=head, graph_uuid=self.run_uuid, batch_time=batch_time, 
                batch_idx=batch_idx, source_chunks=source_chunks
            )
            
            # Create tail entity
            await session.run(
                """
                MERGE (h:Entity:GraphNode {name: $tail, graph_uuid: $graph_uuid})
                SET h.created_at = datetime(),
                    h.batch_time = $batch_time,
                    h.batch_id = $batch_idx,
                    h.source_chunks = $source_chunks
                """,
                tail=tail, graph_uuid=self.run_uuid, batch_time=batch_time,
                batch_idx=batch_idx, source_chunks=source_chunks
            )
            
            # Create relationship
            await session.run(
                f"""
                MATCH (h:Entity {{name: $head, graph_uuid: $graph_uuid}})
                MATCH (t:Entity {{name: $tail, graph_uuid: $graph_uuid}})
                MERGE (h)-[r:`{relation.replace(' ', '_').upper()}` {{graph_uuid: $graph_uuid}}]->(t)
                SET r.source_chunks = $source_chunks,
                    r.batch_id = $batch_idx
                """,
                head=head, tail=tail, graph_uuid=self.run_uuid, source_chunks=source_chunks, batch_idx=batch_idx
            )

    async def _create_chunks_with_embeddings(self, session, batch_data: List[Dict], triplets: List[Dict[str, Any]], batch_idx: int = 0, text_chunks: List[Dict[str, Any]] = None):
        """Create Chunk nodes with embeddings for hybrid vector+entity retrieval"""
        
        # 1. Use text_chunks if provided (New Logic: ID-based)
        if text_chunks:
            for chunk in text_chunks:
                chunk_id = chunk["id"]
                content = chunk["content"]
                embedding = chunk.get("embedding")
                chunk_index = chunk.get("index") if (isinstance(chunk.get("index"), int) or isinstance(chunk.get("index"), str)) else None
                
                # Create chunk node
                if embedding:
                    await session.run(
                        """
                        MERGE (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                        SET c.content = $content,
                            c.embedding = $embedding,
                            c.created_at = datetime(),
                            c.batch_id = $batch_idx,
                            c.embedding_model = $embedding_model
                        """,
                        chunk_id=chunk_id, graph_uuid=self.run_uuid,
                        content=content, embedding=embedding, batch_idx=batch_idx,
                        embedding_model=self.kg_config.embedding_model
                    )
                else:
                    await session.run(
                        """
                        MERGE (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                        SET c.content = $content,
                            c.created_at = datetime(),
                            c.batch_id = $batch_idx
                        """,
                        chunk_id=chunk_id, graph_uuid=self.run_uuid,
                        content=content, batch_idx=batch_idx
                    )
                
                # Link entities to this chunk
                # source_chunks in triplets may be full chunk IDs (uuid_batch_chunk) or short ids (batch_chunk)
                for triplet in triplets:
                    source_chunks = triplet.get('source_chunks', []) or []
                    # Normalize to strings for comparison
                    source_chunks_str = [str(s) for s in source_chunks]

                    # Derive short id for the current chunk (e.g., '0_3' from '..._0_3')
                    parts = str(chunk_id).split('_')
                    short_id = None
                    if len(parts) >= 2:
                        short_id = f"{parts[-2]}_{parts[-1]}"

                    # Match if either the full chunk_id or the short form appears in the triplet's source list
                    matched = False
                    # 1) Full chunk_id match (uuid_batch_idx)
                    if chunk_id in source_chunks_str:
                        matched = True
                    # 2) Short form match (batch_idx_chunk_idx)
                    elif short_id and short_id in source_chunks_str:
                        matched = True
                    # 3) direct numeric/index match (triplets may use integer index or string of index)
                    elif chunk_index is not None and (str(chunk_index) in source_chunks_str or chunk_index in source_chunks):
                        matched = True
                    # 4) If a source chunk contains a trailing short id part (e.g., runuuid_0_3 vs 0_3), try suffix match
                    elif any(sc.endswith(f"_{chunk_index}") or sc.endswith(f"_{short_id}") for sc in source_chunks_str if sc):
                        matched = True

                    if not matched:
                        continue

                    head = triplet.get('head', '')
                    tail = triplet.get('tail', '')

                    if head:
                        await session.run(
                            """
                            MATCH (e:Entity {name: $entity, graph_uuid: $graph_uuid})
                            MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                            MERGE (e)-[:FROM_CHUNK]->(c)
                            """,
                            entity=head, chunk_id=chunk_id, graph_uuid=self.run_uuid
                        )
                    if tail:
                        await session.run(
                            """
                            MATCH (e:Entity {name: $entity, graph_uuid: $graph_uuid})
                            MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                            MERGE (e)-[:FROM_CHUNK]->(c)
                            """,
                            entity=tail, chunk_id=chunk_id, graph_uuid=self.run_uuid
                        )

            # Update entity source_chunk_ids
            try:
                await session.run(
                    """
                    MATCH (e:Entity:GraphNode)-[:FROM_CHUNK]->(c:Chunk:GraphNode)
                    WHERE e.graph_uuid = $graph_uuid AND c.graph_uuid = $graph_uuid AND c.batch_id = $batch_idx
                    WITH e, collect(DISTINCT c.id) AS new_chunk_ids
                    SET e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + new_chunk_ids
                    """,
                    graph_uuid=self.run_uuid, batch_idx=batch_idx
                )
            except Exception as e:
                logger.warning(f"Failed to update entity source_chunk_ids for batch {batch_idx}: {e}")
            
            return

        # 2. Fallback to original logic (VLM-based chunks)
        if not batch_data:
            return
        
        try:
            # Generate embeddings asynchronously for all chunks
            embedding_tasks = []
            for chunk_data in batch_data:
                content = chunk_data.get('content', '')
                task = asyncio.create_task(self.embedder.aembed_query(content)) if content else None
                embedding_tasks.append((chunk_data, task))
            
            # Create chunk nodes with embeddings
            for chunk_data, embedding_task in embedding_tasks:
                chunk_time = chunk_data.get('time', '')
                content = chunk_data.get('content', '')
                chunk_id = f"{self.run_uuid}_{chunk_time}".replace(' ', '_').replace(':', '')
                
                # Get embedding if task exists
                embedding = None
                if embedding_task:
                    try:
                        embedding = await embedding_task
                    except Exception as e:
                        logger.warning(f"Failed to embed chunk at {chunk_time}: {e}")
                
                # Create chunk node with optional embedding
                if embedding:
                    await session.run(
                        """
                        MERGE (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                        SET c.time = $time,
                            c.content = $content,
                            c.embedding = $embedding,
                            c.created_at = datetime(),
                            c.batch_id = $batch_idx,
                            c.embedding_model = $embedding_model
                        """,
                        chunk_id=chunk_id, graph_uuid=self.run_uuid, time=chunk_time,
                        content=content, embedding=embedding, batch_idx=batch_idx,
                        embedding_model=self.kg_config.embedding_model
                    )
                else:
                    await session.run(
                        """
                        MERGE (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                        SET c.time = $time,
                            c.content = $content,
                            c.created_at = datetime(),
                            c.batch_id = $batch_idx
                        """,
                        chunk_id=chunk_id, graph_uuid=self.run_uuid, time=chunk_time,
                        content=content, batch_idx=batch_idx
                    )
                
                # Create relationships from entities to chunks based on source_chunks
                for triplet in triplets:
                    source_chunks = triplet.get('source_chunks', [])
                    # Find which batch index this chunk is (by time matching)
                    chunk_idx = next((i for i, bd in enumerate(batch_data) if bd.get('time') == chunk_time), -1)
                    
                    if chunk_idx in source_chunks:
                        # Link entities from this triplet to this chunk
                        head = triplet.get('head', '')
                        tail = triplet.get('tail', '')
                        
                        if head:
                            await session.run(
                                """
                                MATCH (e:Entity {name: $entity, graph_uuid: $graph_uuid})
                                MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                                MERGE (e)-[:FROM_CHUNK]->(c)
                                """,
                                entity=head, chunk_id=chunk_id, graph_uuid=self.run_uuid
                            )
                        
                        if tail:
                            await session.run(
                                """
                                MATCH (e:Entity {name: $entity, graph_uuid: $graph_uuid})
                                MATCH (c:Chunk {id: $chunk_id, graph_uuid: $graph_uuid})
                                MERGE (e)-[:FROM_CHUNK]->(c)
                                """,
                                entity=tail, chunk_id=chunk_id, graph_uuid=self.run_uuid
                            )
        
                # After creating chunk nodes & FROM_CHUNK relationships for this batch,
                # propagate the list of chunk IDs to each entity as `source_chunk_ids` property
                try:
                    await session.run(
                        """
                        MATCH (e:Entity:GraphNode)-[:FROM_CHUNK]->(c:Chunk:GraphNode)
                        WHERE e.graph_uuid = $graph_uuid AND c.graph_uuid = $graph_uuid AND c.batch_id = $batch_idx
                        WITH e, collect(DISTINCT c.id) AS new_chunk_ids
                        SET e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + new_chunk_ids
                        """,
                        graph_uuid=self.run_uuid, batch_idx=batch_idx
                    )
                except Exception as e:
                    logger.warning(f"Failed to update entity source_chunk_ids for batch {batch_idx}: {e}")
        except Exception as e:
            logger.error(f"Error creating chunks with embeddings: {str(e)}")
            # Fallback: create chunks without embeddings
            for chunk_data in batch_data:
                chunk_time = chunk_data.get('time', '')
                content = chunk_data.get('content', '')
                chunk_id = f"{self.run_uuid}_{chunk_time}".replace(' ', '_').replace(':', '')
                
                await session.run(
                    """
                    MERGE (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                    SET c.time = $time,
                        c.content = $content,
                        c.created_at = datetime()
                    """,
                    chunk_id=chunk_id, graph_uuid=self.run_uuid, time=chunk_time,
                    content=content
                )

    async def _ensure_indexes(self, session):
        """Ensure needed indexes exist - create them if missing"""
        try:
            await session.execute_write(self._create_indexes)
            logger.info("Verified/created graph indexes")
        except Exception as e:
            logger.warning(f"Index verification/creation warning: {e}")

    async def _apply_operations(self, session, operations: Dict[str, Any], batch_idx: int = 0):
        """Apply merge, inter_chunk_relations, and prune operations in the recommended order.

        Operations dict expected keys: new_triplets, inter_chunk_relations, merge_instructions, prune_instructions
        """
        if not operations:
            return

        merges = operations.get('merge_instructions') or []
        inter_links = operations.get('inter_chunk_relations') or []
        prunes = operations.get('prune_instructions') or []

        # 1) Merges first (to consolidate entities before linking/pruning)
        if merges:
            logger.info(f"Applying {len(merges)} merge instructions")
            for m in merges:
                local = m.get('local')
                existing = m.get('existing')
                existing_id = m.get('existing_id')  # Not used in query, but logged
                if local == existing:  # Skip self-merges
                    logger.debug(f"Skipping self-merge: {local} -> {existing}")
                    continue
                try:
                    # Try APOC merge if available (best effort)
                    apoc_query = """
                    MATCH (l:Entity:GraphNode {graph_uuid: $graph_uuid})
                    WHERE l.name = $local OR $local IN l.name
                    MATCH (e:Entity:GraphNode {graph_uuid: $graph_uuid})
                    WHERE e.name = $existing OR $existing IN e.name
                    CALL apoc.refactor.mergeNodes([l, e], {properties: {name: $existing}}) YIELD node
                    SET node.name = $existing
                    RETURN node
                    """
                    await session.run(apoc_query, local=local, existing=existing, graph_uuid=self.run_uuid)
                    logger.debug(f"APOC merge successful: {local} -> {existing}")
                except Exception as apoc_err:
                    logger.debug(f"APOC merge failed for {local} -> {existing}: {apoc_err}")
                    # Fallback: Manual merge with fixed Cypher structure
                    try:
                        fallback_query = """
                        // Match local and existing
                        MATCH (l:Entity:GraphNode {name: $local, graph_uuid: $graph_uuid})
                        MATCH (e:Entity:GraphNode {name: $existing, graph_uuid: $graph_uuid})
                        
                        // Transfer relationships (outgoing from local to existing)
                        MATCH (l)-[rel_out:GraphRel]->(target)
                        WHERE rel_out.graph_uuid = $graph_uuid
                        MERGE (e)-[:`{rel_type}` {{graph_uuid: $graph_uuid}}]->(target)
                        SET {props}
                        DELETE rel_out
                        
                        // Transfer incoming relationships (to local from source to existing)
                        MATCH (source)-[rel_in:GraphRel]-(l)
                        WHERE rel_in.graph_uuid = $graph_uuid
                        MERGE (source)-[:`{rel_type_in}` {{graph_uuid: $graph_uuid}}]-(e)
                        SET {props_in}
                        DELETE rel_in
                        
                        // Transfer FROM_CHUNK relationships
                        MATCH (l)-[:FROM_CHUNK]->(c:Chunk:GraphNode)
                        MERGE (e)-[:FROM_CHUNK]->(c)
                        WITH e, collect(DISTINCT c.id) AS new_chunks
                        SET e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + new_chunks
                        
                        // Combine properties (excluding id/name)
                        WITH l, e
                        SET e += apoc.map.removeKey(l, 'name'),
                            e.updated_at = datetime()
                        
                        // Delete local node
                        DETACH DELETE l
                        """
                        # Note: This is a simplified version; for full rel transfer, we'd need dynamic rel types.
                        # For now, use a batched approach or log for manual review.
                        simple_transfer_query = """
                        MATCH (l:Entity:GraphNode {graph_uuid: $graph_uuid})
                        WHERE l.name = $local OR $local IN l.name
                        MATCH (e:Entity:GraphNode {graph_uuid: $graph_uuid})
                        WHERE e.name = $existing OR $existing IN e.name
                        
                        // Transfer FROM_CHUNK
                        MATCH (l)-[f:FROM_CHUNK]->(c:Chunk:GraphNode)
                        CREATE (e)-[:FROM_CHUNK]->(c)
                        DELETE f
                        
                        // Aggregate and set chunk ids on existing
                        WITH e
                        OPTIONAL MATCH (l)-[:FROM_CHUNK]->(c2:Chunk:GraphNode)
                        WITH e, collect(DISTINCT c2.id) AS new_chunks
                        SET e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + new_chunks
                        
                        // Combine scalar properties (add timestamp for tracking)
                        WITH l, e
                        SET e += l {
                            .*, 
                            name: $existing, 
                            merged_from: coalesce(e.merged_from, []) + $local
                        },
                        e.updated_at = datetime()
                        
                        // Delete local (after transfer)
                        DETACH DELETE l
                        """
                        await session.run(simple_transfer_query, local=local, existing=existing, graph_uuid=self.run_uuid)
                        logger.debug(f"Fallback merge successful: {local} -> {existing}")
                    except Exception as fallback_err:
                        logger.warning(f"Fallback merge failed for {local} -> {existing}: {fallback_err}")
                        # Ultimate fallback: Just delete local if isolated
                        try:
                            await session.run(
                                "MATCH (l:Entity:GraphNode {graph_uuid: $graph_uuid}) WHERE l.name = $local OR $local IN l.name DETACH DELETE l",
                                local=local, graph_uuid=self.run_uuid
                            )
                            logger.info(f"Force-deleted isolated local entity: {local}")
                        except Exception as force_err:
                            logger.error(f"Even force-delete failed for {local}: {force_err}")

        # 2) Prunes (now after merges, to target correct entities)
        if prunes:
            logger.info(f"Applying {len(prunes)} prune instructions")
            for p in prunes:
                try:
                    # Check if this is an entity prune or relationship prune
                    if 'entity' in p:
                        # Entity prune: delete entity and all its relationships
                        entity_name = p.get('entity')
                        if not entity_name:
                            continue
                        
                        result = await session.run(
                            """
                            MATCH (n:Entity:GraphNode {graph_uuid: $graph_uuid})
                            WHERE n.name = $entity_name OR $entity_name IN n.name
                            DETACH DELETE n
                            RETURN count(n) as deleted
                            """,
                            entity_name=entity_name, graph_uuid=self.run_uuid
                        )
                        record = await result.single()
                        deleted = record.get('deleted', 0) if record else 0
                        if deleted > 0:
                            logger.debug(f"Pruned entity and all its relationships: {entity_name}")
                        else:
                            logger.debug(f"No matching entity found to prune: {entity_name}")
                    
                    elif 'head' in p and 'relation' in p and 'tail' in p:
                        # Relationship prune: delete specific relationship
                        head = p.get('head')
                        rel = p.get('relation')
                        tail = p.get('tail')
                        if not head or not rel or not tail:
                            continue
                        
                        rel_label = rel.replace(' ', '_').upper()
                        result = await session.run(
                            f"""
                            MATCH (h:Entity:GraphNode {{graph_uuid: $graph_uuid}})
                            WHERE h.name = $head OR $head IN h.name
                            MATCH (t:Entity:GraphNode {{graph_uuid: $graph_uuid}})
                            WHERE t.name = $tail OR $tail IN t.name
                            MATCH (h)-[r:`{rel_label}`]-(t)
                            DETACH DELETE r
                            RETURN count(r) as deleted
                            """,
                            head=head, tail=tail, graph_uuid=self.run_uuid
                        )
                        record = await result.single()
                        deleted = record.get('deleted', 0) if record else 0
                        if deleted > 0:
                            logger.debug(f"Pruned relationship: {head} -[{rel}]-> {tail}")
                        else:
                            logger.debug(f"No matching relationship found to prune: {head} -[{rel}]-> {tail}")
                    
                    else:
                        logger.warning(f"Invalid prune instruction format: {p}")
                        
                except Exception as e:
                    logger.warning(f"Failed to prune {p}: {e}")

        # 3) Inter-chunk relations (last, after cleaning)
        if inter_links:
            logger.info(f"Applying {len(inter_links)} inter-chunk relations")
            for it in inter_links:
                try:
                    if len(it) < 3:
                        continue
                    head = it[0]
                    rel = it[1]
                    tail = it[2]
                    source_chunks = it[3] if len(it) > 3 and isinstance(it[3], list) else []
                    rel_label = rel.replace(' ', '_').upper()
                    query = f"""
                    MATCH (h:Entity:GraphNode {{graph_uuid: $graph_uuid}})
                    WHERE h.name = $head OR $head IN h.name
                    MATCH (t:Entity:GraphNode {{graph_uuid: $graph_uuid}})
                    WHERE t.name = $tail OR $tail IN t.name
                    MERGE (h)-[r:`{rel_label}` {{graph_uuid: $graph_uuid}}]->(t)
                    SET r.source_chunks = coalesce(r.source_chunks, []) + $source_chunks,
                        r.batch_id = $batch_idx
                    RETURN count(r) as created
                    """
                    result = await session.run(query, head=head, tail=tail, graph_uuid=self.run_uuid, source_chunks=source_chunks, batch_idx=batch_idx)
                    record = await result.single()
                    created = record.get('created', 0) if record else 0
                    if created > 0:
                        logger.debug(f"Created inter-chunk relation: {head} -[{rel}]-> {tail}")
                    else:
                        logger.debug(f"No inter-chunk relation created: {head} -[{rel}]-> {tail}")
                except Exception as e:
                    logger.warning(f"Failed to create inter-chunk relation {it}: {e}")

        # Post-cleanup: Optional aggressive cleanup (e.g., remove isolated nodes with degree < 1)
        await self._cleanup_isolated_nodes(session)

    async def _cleanup_isolated_nodes(self, session):
        """Aggressively remove isolated nodes (degree 0) to clean the graph"""
        try:
            # Delete isolated Entity nodes (no relationships)
            result = await session.run("""
                MATCH (n:Entity:GraphNode)
                WHERE n.graph_uuid = $graph_uuid
                AND NOT (n)--()
                DETACH DELETE n
                RETURN count(n) as deleted
            """, graph_uuid=self.run_uuid)
            record = await result.single()
            deleted_entities = record.get('deleted', 0) if record else 0
            if deleted_entities > 0:
                logger.info(f"Cleaned {deleted_entities} isolated Entity nodes")
            # NOTE: We intentionally DO NOT delete isolated Chunk nodes here.
            # Chunks may be generated independently (e.g., by VLM or external chunker)
            # and can still be useful for vector-based retrieval even if no
            # FROM_CHUNK relationships were created (or created using different
            # id formats). Deleting isolated chunks caused hybrid retrieval to
            # return no chunk seeds; keep them to ensure vector search works.
            logger.debug("Skipping deletion of isolated Chunk nodes to preserve vector-searchable chunks")
        except Exception as e:
            logger.warning(f"Graph cleanup failed: {e}")

    async def get_node_count(self) -> int:
        """Get the count of nodes in the current graph run"""
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH (n:GraphNode) WHERE n.graph_uuid = $graph_uuid RETURN count(n) as count",
                    graph_uuid=self.run_uuid
                )
                record = await result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.error(f"Error getting node count: {str(e)}")
            return 0

    async def get_chunk_counts(self) -> Dict[str, int]:
        """Return a breakdown of chunks in the current run: total, with embeddings, and linked chunks"""
        try:
            async with self.driver.session() as session:
                total_res = await session.run(
                    "MATCH (c:Chunk:GraphNode) WHERE c.graph_uuid = $graph_uuid RETURN count(c) as count",
                    graph_uuid=self.run_uuid
                )
                total = (await total_res.single())["count"] if total_res else 0

                emb_res = await session.run(
                    "MATCH (c:Chunk:GraphNode) WHERE c.graph_uuid = $graph_uuid AND c.embedding IS NOT NULL RETURN count(c) as count",
                    graph_uuid=self.run_uuid
                )
                with_emb = (await emb_res.single())["count"] if emb_res else 0

                linked_res = await session.run(
                    "MATCH (c:Chunk:GraphNode)<-[:FROM_CHUNK]-(:Entity) WHERE c.graph_uuid = $graph_uuid RETURN count(DISTINCT c) as count",
                    graph_uuid=self.run_uuid
                )
                linked = (await linked_res.single())["count"] if linked_res else 0

                return {"total_chunks": int(total), "with_embedding": int(with_emb), "linked_chunks": int(linked)}
        except Exception as e:
            logger.warning(f"Failed to get chunk counts: {e}")
            return {"total_chunks": 0, "with_embedding": 0, "linked_chunks": 0}

    async def get_relationship_count(self) -> int:
        """Get the count of relationships in the current graph run"""
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH ()-[r]->() WHERE r.graph_uuid = $graph_uuid RETURN count(r) as count",
                    graph_uuid=self.run_uuid
                )
                record = await result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.error(f"Error getting relationship count: {str(e)}")
            return 0

    # ========== Community High Graph Methods ==========

    async def create_chunk_questions(self, chunk_questions: Dict[str, List[str]], batch_idx: int = 0) -> int:
        """
        Create ChunkQuestion nodes and link them to their parent Chunk nodes.
        
        Args:
            chunk_questions: Dict mapping chunk_id -> list of questions
            batch_idx: Current batch index for tracking
            
        Returns:
            Number of questions created
        """
        if not chunk_questions:
            return 0
        
        created_count = 0
        try:
            async with self.driver.session() as session:
                for chunk_id, questions in chunk_questions.items():
                    for q_idx, question in enumerate(questions):
                        question_id = f"{chunk_id}_q{q_idx}"
                        
                        # Create ChunkQuestion node and link to parent Chunk
                        await session.run(
                            """
                            MATCH (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                            MERGE (q:ChunkQuestion:GraphNode {id: $question_id, graph_uuid: $graph_uuid})
                            SET q.question = $question,
                                q.chunk_id = $chunk_id,
                                q.batch_id = $batch_idx,
                                q.created_at = datetime()
                            MERGE (c)-[:HAS_QUESTION]->(q)
                            """,
                            chunk_id=chunk_id,
                            question_id=question_id,
                            question=question,
                            batch_idx=batch_idx,
                            graph_uuid=self.run_uuid
                        )
                        created_count += 1
                
                logger.info(f"Created {created_count} ChunkQuestion nodes across {len(chunk_questions)} chunks")
                return created_count
        except Exception as e:
            logger.error(f"Error creating chunk questions: {e}")
            return created_count

    async def get_all_chunks_with_questions(self) -> List[Dict[str, Any]]:
        """
        Get all Chunk nodes with their associated questions.
        
        Returns:
            List of dicts with chunk_id, content, questions, and community_id (if set)
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Chunk:GraphNode {graph_uuid: $graph_uuid})
                    OPTIONAL MATCH (c)-[:HAS_QUESTION]->(q:ChunkQuestion)
                    WITH c, collect(q.question) as questions
                    RETURN c.id as chunk_id, c.content as content, 
                           c.community_id as community_id, questions
                    """,
                    graph_uuid=self.run_uuid
                )
                
                chunks = []
                async for record in result:
                    chunks.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "community_id": record["community_id"],
                        "questions": record["questions"] or []
                    })
                
                return chunks
        except Exception as e:
            logger.error(f"Error getting chunks with questions: {e}")
            return []

    async def update_chunk_community_ids(self, chunk_community_map: Dict[str, str]) -> int:
        """
        Update community_id property on Chunk nodes.
        
        Args:
            chunk_community_map: Dict mapping chunk_id -> community_id (stable hash)
            
        Returns:
            Number of chunks updated
        """
        if not chunk_community_map:
            return 0
        
        updated_count = 0
        try:
            async with self.driver.session() as session:
                for chunk_id, community_id in chunk_community_map.items():
                    result = await session.run(
                        """
                        MATCH (c:Chunk:GraphNode {id: $chunk_id, graph_uuid: $graph_uuid})
                        SET c.community_id = $community_id,
                            c.prev_community_id = c.community_id
                        RETURN count(c) as updated
                        """,
                        chunk_id=chunk_id,
                        community_id=community_id,
                        graph_uuid=self.run_uuid
                    )
                    record = await result.single()
                    if record and record["updated"] > 0:
                        updated_count += 1
                
                logger.debug(f"Updated community_id on {updated_count} chunks")
                return updated_count
        except Exception as e:
            logger.error(f"Error updating chunk community IDs: {e}")
            return updated_count

    async def upsert_community_summary(self, community_id: str, content: str, embedding: List[float], 
                                       member_chunk_ids: List[str], 
                                       question_chunks: List[str] = None,
                                       embedding_chunks: List[List[float]] = None) -> bool:
        """
        Create or update a CommunitySummary node with chunked embeddings.
        
        Args:
            community_id: Stable hash ID for the community
            content: Full concatenated questions from member chunks
            embedding: Primary vector embedding (first chunk or full content)
            member_chunk_ids: List of chunk IDs belonging to this community
            question_chunks: List of question text chunks (for sparse-to-dense)
            embedding_chunks: List of embeddings, one per question_chunk (for MaxSim retrieval)
            
        Returns:
            True if successful
        """
        try:
            import json
            # Normalize embedding_chunks to a list of JSON strings, because Neo4j properties do not support nested lists
            if embedding_chunks is None:
                embedding_chunks_serialized = [json.dumps(embedding)]
            else:
                embedding_chunks_serialized = []
                for emb in embedding_chunks:
                    # If already a string, assume it's serialized
                    if isinstance(emb, str):
                        embedding_chunks_serialized.append(emb)
                    else:
                        # Convert numpy arrays or other iterables to list of floats
                        try:
                            emb_list = [float(x) for x in emb]
                        except Exception:
                            emb_list = list(emb)
                        embedding_chunks_serialized.append(json.dumps(emb_list))

            async with self.driver.session() as session:
                await session.run(
                    """
                    MERGE (cs:CommunitySummary:GraphNode {community_id: $community_id, graph_uuid: $graph_uuid})
                    SET cs.content = $content,
                        cs.embedding = $embedding,
                        cs.member_chunk_ids = $member_chunk_ids,
                        cs.question_chunks = $question_chunks,
                        cs.embedding_chunks = $embedding_chunks,
                        cs.num_embedding_chunks = $num_chunks,
                        cs.updated_at = datetime()
                    """,
                    community_id=community_id,
                    content=content,
                    embedding=embedding,
                    member_chunk_ids=member_chunk_ids,
                    question_chunks=question_chunks or [content],
                    embedding_chunks=embedding_chunks_serialized,
                    num_chunks=len(embedding_chunks_serialized) if embedding_chunks_serialized else 1,
                    graph_uuid=self.run_uuid
                )
                return True
        except Exception as e:
            logger.error(f"Error upserting community summary {community_id}: {e}")
            return False

    async def get_all_community_summaries(self) -> List[Dict[str, Any]]:
        """
        Get all CommunitySummary nodes for the current graph.
        
        Returns:
            List of dicts with community_id, content, embedding, member_chunk_ids
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (cs:CommunitySummary:GraphNode {graph_uuid: $graph_uuid})
                    RETURN cs.community_id as community_id, cs.content as content,
                           cs.embedding as embedding, cs.member_chunk_ids as member_chunk_ids,
                           cs.embedding_chunks as embedding_chunks, cs.num_embedding_chunks as num_chunks
                    """,
                    graph_uuid=self.run_uuid
                )
                
                summaries = []
                import json
                async for record in result:
                    raw_chunks = record.get("embedding_chunks")
                    parsed_chunks = []
                    if raw_chunks:
                        for item in raw_chunks:
                            if isinstance(item, str):
                                try:
                                    parsed_chunks.append(json.loads(item))
                                except Exception:
                                    continue
                            elif isinstance(item, list):
                                parsed_chunks.append(item)

                    summaries.append({
                        "community_id": record["community_id"],
                        "content": record["content"],
                        "embedding": record["embedding"],
                        "member_chunk_ids": record["member_chunk_ids"] or [],
                        "embedding_chunks": parsed_chunks,
                        "num_chunks": record.get("num_chunks", 1)
                    })
                
                return summaries
        except Exception as e:
            logger.error(f"Error getting community summaries: {e}")
            return []

    async def delete_orphan_community_summaries(self, valid_community_ids: List[str]) -> int:
        """
        Delete CommunitySummary nodes whose community_id is no longer valid.
        
        Args:
            valid_community_ids: List of currently valid community IDs
            
        Returns:
            Number of summaries deleted
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (cs:CommunitySummary:GraphNode {graph_uuid: $graph_uuid})
                    WHERE NOT cs.community_id IN $valid_ids
                    DETACH DELETE cs
                    RETURN count(cs) as deleted
                    """,
                    valid_ids=valid_community_ids,
                    graph_uuid=self.run_uuid
                )
                record = await result.single()
                deleted = record["deleted"] if record else 0
                if deleted > 0:
                    logger.info(f"Deleted {deleted} orphan CommunitySummary nodes")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting orphan community summaries: {e}")
            return 0

    async def get_chunks_in_community(self, community_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks belonging to a specific community.
        
        Args:
            community_id: The community ID to query
            
        Returns:
            List of chunk dicts with id, content, questions
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Chunk:GraphNode {graph_uuid: $graph_uuid, community_id: $community_id})
                    OPTIONAL MATCH (c)-[:HAS_QUESTION]->(q:ChunkQuestion)
                    WITH c, collect(q.question) as questions
                    RETURN c.id as chunk_id, c.content as content, questions
                    """,
                    graph_uuid=self.run_uuid,
                    community_id=community_id
                )
                
                chunks = []
                async for record in result:
                    chunks.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "questions": record["questions"] or []
                    })
                
                return chunks
        except Exception as e:
            logger.error(f"Error getting chunks in community {community_id}: {e}")
            return []

    async def vector_search_community_summaries(self, query_embedding: List[float], top_k: int = 5, 
                                                min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Vector similarity search on CommunitySummary node embeddings using MaxSim scoring.
        
        For communities with multiple embedding chunks (sparse-to-dense), computes similarity
        against each chunk and takes the maximum (ColBERT-style MaxSim).
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of community summaries with similarity scores and member chunk ids
        """
        try:
            async with self.driver.session() as session:
                # Get all communities with their embedding chunks
                result = await session.run(
                    """
                    MATCH (cs:CommunitySummary:GraphNode {graph_uuid: $graph_uuid})
                    WHERE cs.embedding IS NOT NULL
                    RETURN cs.community_id AS community_id, 
                           cs.content AS content, 
                           cs.member_chunk_ids AS member_chunk_ids,
                           cs.embedding AS embedding,
                           cs.embedding_chunks AS embedding_chunks,
                           cs.num_embedding_chunks AS num_chunks
                    """,
                    graph_uuid=self.run_uuid
                )
                
                # Compute MaxSim scores in Python for flexibility
                import math
                import json

                def cosine_sim(v1, v2):
                    dot = sum(a * b for a, b in zip(v1, v2))
                    mag1 = math.sqrt(sum(a * a for a in v1))
                    mag2 = math.sqrt(sum(b * b for b in v2))
                    if mag1 * mag2 == 0:
                        return 0.0
                    return dot / (mag1 * mag2)
                
                summaries = []
                async for record in result:
                    raw_chunks = record.get("embedding_chunks")
                    embedding_chunks = []
                    if raw_chunks:
                        # raw_chunks can be stored as JSON strings or lists depending on node version
                        for item in raw_chunks:
                            if isinstance(item, str):
                                try:
                                    emb = json.loads(item)
                                except Exception:
                                    # If parsing fails, skip the chunk
                                    continue
                            else:
                                emb = item
                            # Ensure it's a numeric vector
                            if isinstance(emb, list) and emb:
                                embedding_chunks.append(emb)

                    # MaxSim: take max similarity across all embedding chunks
                    if embedding_chunks:
                        max_sim = max(cosine_sim(query_embedding, emb) for emb in embedding_chunks)
                    else:
                        # Fallback to primary embedding
                        max_sim = cosine_sim(query_embedding, record["embedding"])
                    
                    if max_sim > min_similarity:
                        summaries.append({
                            "community_id": record["community_id"],
                            "content": record["content"],
                            "member_chunk_ids": record["member_chunk_ids"] or [],
                            "score": float(max_sim),
                            "num_chunks": record.get("num_chunks", 1),
                            "source": "community_maxsim"
                        })
                
                # Sort by score and limit
                summaries.sort(key=lambda x: x["score"], reverse=True)
                summaries = summaries[:top_k]
                
                logger.debug(f"Community MaxSim search returned {len(summaries)} summaries")
                return summaries
        except Exception as e:
            logger.error(f"Error in community vector search: {e}")
            return []

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of chunk dicts with id, content, time
        """
        if not chunk_ids:
            return []
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Chunk:GraphNode {graph_uuid: $graph_uuid})
                    WHERE c.id IN $chunk_ids
                    RETURN c.id AS chunk_id, c.content AS content, c.time AS chunk_time
                    """,
                    graph_uuid=self.run_uuid,
                    chunk_ids=chunk_ids
                )
                
                chunks = []
                async for record in result:
                    chunks.append({
                        "id": record["chunk_id"],
                        "content": record["content"],
                        "time": record["chunk_time"],
                        "source": "community_hop"
                    })
                
                return chunks
        except Exception as e:
            logger.error(f"Error getting chunks by IDs: {e}")
            return []

    async def get_entities_from_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get all entities connected to the given chunks.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of entity dicts with name, batch_time
        """
        if not chunk_ids:
            return []
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Chunk:GraphNode {graph_uuid: $graph_uuid})-[:CONTAINS]->(e:Entity:GraphNode)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.name AS name, e.batch_time AS batch_time
                    """,
                    graph_uuid=self.run_uuid,
                    chunk_ids=chunk_ids
                )
                
                entities = []
                async for record in result:
                    entities.append({
                        "name": record["name"],
                        "batch_time": record["batch_time"] or "",
                        "source": "community_hop"
                    })
                
                return entities
        except Exception as e:
            logger.error(f"Error getting entities from chunks: {e}")
            return []

    # ========== Pre-retrieval Computation Methods ==========

    async def graph_exists(self, graph_uuid: str) -> bool:
        """
        Check if a graph with the given UUID exists.
        
        Args:
            graph_uuid: UUID of the graph to check
            
        Returns:
            True if graph exists, False otherwise
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (n {graph_uuid: $graph_uuid})
                    RETURN count(n) > 0 AS exists
                    LIMIT 1
                    """,
                    graph_uuid=graph_uuid
                )
                record = await result.single()
                return record["exists"] if record else False
        except Exception as e:
            logger.error(f"Error checking graph existence: {e}")
            return False

    async def set_precomputation_flag(self, graph_uuid: str, method: str, value: bool) -> None:
        """
        Set a flag indicating that precomputation has been performed for a method.
        Stores this as a property on a special GraphMetadata node.
        
        Args:
            graph_uuid: UUID of the graph
            method: Precomputation method ('page_rank', 'ch3_l3')
            value: True if precomputation is complete, False otherwise
        """
        try:
            async with self.driver.session() as session:
                # Create or update GraphMetadata node
                await session.run(
                    """
                    MERGE (m:GraphMetadata:GraphNode {graph_uuid: $graph_uuid})
                    SET m[$method_flag] = $value
                    """,
                    graph_uuid=graph_uuid,
                    method_flag=f"precomputed_{method}",
                    value=value
                )
                logger.info(f"Set precomputation flag for {method} = {value}")
        except Exception as e:
            logger.error(f"Error setting precomputation flag: {e}")

    async def has_precomputation(self, graph_uuid: str, method: str) -> bool:
        """
        Check if precomputation has been performed for a given method.
        
        Args:
            graph_uuid: UUID of the graph
            method: Precomputation method ('page_rank', 'ch3_l3')
            
        Returns:
            True if precomputation exists, False otherwise
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (m:GraphMetadata:GraphNode {graph_uuid: $graph_uuid})
                    RETURN m[$method_flag] AS has_precompute
                    """,
                    graph_uuid=graph_uuid,
                    method_flag=f"precomputed_{method}"
                )
                record = await result.single()
                return bool(record["has_precompute"]) if record else False
        except Exception as e:
            logger.error(f"Error checking precomputation flag: {e}")
            return False

    async def get_ppr_top_neighbors(self, entity_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-K neighbors for an entity based on precomputed PPR scores.
        
        Args:
            entity_name: Name of the entity
            top_k: Number of neighbors to return
            
        Returns:
            List of dicts with 'node' and 'score' keys
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {name: $name, graph_uuid: $graph_uuid})
                    WHERE e.ppr_top_neighbors IS NOT NULL
                    RETURN e.ppr_top_neighbors AS scores
                    """,
                    name=entity_name,
                    graph_uuid=self.run_uuid
                )
                record = await result.single()
                
                if not record or not record["scores"]:
                    return []
                
                # Parse JSON scores
                scores = json.loads(record["scores"])
                return scores[:top_k]
                
        except Exception as e:
            logger.warning(f"Error getting PPR neighbors for {entity_name}: {e}")
            return []

    async def get_ch3l3_top_candidates(self, entity_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top-K candidates for an entity based on precomputed CH3-L3 scores.
        
        Args:
            entity_name: Name of the entity
            top_k: Number of candidates to return
            
        Returns:
            List of dicts with 'node' and 'score' keys
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {name: $name, graph_uuid: $graph_uuid})
                    WHERE e.ch3l3_scores IS NOT NULL
                    RETURN e.ch3l3_scores AS scores
                    """,
                    name=entity_name,
                    graph_uuid=self.run_uuid
                )
                record = await result.single()
                
                if not record or not record["scores"]:
                    return []
                
                # Parse JSON scores
                scores = json.loads(record["scores"])
                return scores[:top_k]
                
        except Exception as e:
            logger.warning(f"Error getting CH3-L3 candidates for {entity_name}: {e}")
            return []

    async def get_chunks_for_entities(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get chunks associated with the given entities via FROM_CHUNK relationships.
        
        Args:
            entity_names: List of entity names
            
        Returns:
            List of chunk dicts with id, content, time
        """
        if not entity_names:
            return []
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {graph_uuid: $graph_uuid})-[:FROM_CHUNK]->(c:Chunk {graph_uuid: $graph_uuid})
                    WHERE e.name IN $entity_names
                    RETURN DISTINCT c.id AS chunk_id, c.content AS content, c.time AS chunk_time
                    """,
                    graph_uuid=self.run_uuid,
                    entity_names=entity_names
                )
                
                chunks = []
                async for record in result:
                    chunks.append({
                        "id": record["chunk_id"],
                        "content": record["content"],
                        "time": record["chunk_time"],
                        "source": "entity_hop"
                    })
                
                return chunks
        except Exception as e:
            logger.error(f"Error getting chunks for entities: {e}")
            return []

    async def get_entity_relationships(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get relationships between the given entities.
        
        Args:
            entity_names: List of entity names
            
        Returns:
            List of relationship dicts with description
        """
        if not entity_names:
            return []
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e1:Entity {graph_uuid: $graph_uuid})-[r]->(e2:Entity {graph_uuid: $graph_uuid})
                    WHERE e1.name IN $entity_names AND e2.name IN $entity_names
                    AND type(r) <> 'FROM_CHUNK'
                    RETURN DISTINCT e1.name AS start_name, type(r) AS rel_type, e2.name AS end_name
                    """,
                    graph_uuid=self.run_uuid,
                    entity_names=entity_names
                )
                
                relationships = []
                async for record in result:
                    desc = f"{record['start_name']} -[{record['rel_type']}]-> {record['end_name']}"
                    relationships.append({
                        "description": desc,
                        "source": "entity_hop"
                    })
                
                return relationships
        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return []

    async def close(self):
        """Close the Neo4j driver connection"""
        await self.driver.close()