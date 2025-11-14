from neo4j import AsyncGraphDatabase, AsyncManagedTransaction
from typing import List, Dict, Any, Optional
import time
import json
from ..core.logger import get_logger
from langchain_openai import OpenAIEmbeddings
import uuid
import logging

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

        # Initialize schema and constraints
        # asyncio.create_task(self._initialize_database())

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

    async def add_batch_to_graph(self, triplets: List[Dict[str, Any]], batch_data: List[Dict]) -> Dict[str, float]:
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
                await self._create_triplets(session, triplets, batch_time)
                
                # Always create chunk nodes for hybrid retrieval
                await self._create_chunks_with_embeddings(session, batch_data, triplets)
                
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

    async def _create_triplets(self, session, triplets: List[Dict[str, Any]], batch_time: str = ""):
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
                    h.source_chunks = $source_chunks
                """,
                head=head, graph_uuid=self.run_uuid, batch_time=batch_time, 
                source_chunks=source_chunks
            )
            
            # Create tail entity
            await session.run(
                """
                MERGE (t:Entity:GraphNode {name: $tail, graph_uuid: $graph_uuid})
                SET t.created_at = datetime(),
                    t.batch_time = $batch_time,
                    t.source_chunks = $source_chunks
                """,
                tail=tail, graph_uuid=self.run_uuid, batch_time=batch_time,
                source_chunks=source_chunks
            )
            
            # Create relationship
            await session.run(
                f"""
                MATCH (h:Entity {{name: $head, graph_uuid: $graph_uuid}})
                MATCH (t:Entity {{name: $tail, graph_uuid: $graph_uuid}})
                MERGE (h)-[r:`{relation.replace(' ', '_').upper()}` {{graph_uuid: $graph_uuid}}]->(t)
                SET r.source_chunks = $source_chunks
                """,
                head=head, tail=tail, graph_uuid=self.run_uuid, source_chunks=source_chunks
            )

    async def _create_chunks_with_embeddings(self, session, batch_data: List[Dict], triplets: List[Dict[str, Any]]):
        """Create Chunk nodes with embeddings for hybrid vector+entity retrieval"""
        if not batch_data:
            return
        
        try:
            # Generate embeddings asynchronously for all chunks
            import asyncio
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
                            c.embedding_model = $embedding_model
                        """,
                        chunk_id=chunk_id, graph_uuid=self.run_uuid, time=chunk_time,
                        content=content, embedding=embedding, 
                        embedding_model=self.kg_config.embedding_model
                    )
                else:
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

    async def close(self):
        """Close the Neo4j driver connection"""
        await self.driver.close()