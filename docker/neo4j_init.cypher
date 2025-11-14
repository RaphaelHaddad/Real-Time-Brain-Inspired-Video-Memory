// Neo4j initialization script for VidGraph
// This script runs once when the container starts for the first time

// Create a basic index for faster lookups
CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE INDEX graph_uuid_index IF NOT EXISTS FOR (n:GraphNode) ON (n.graph_uuid);
CREATE INDEX chunk_time_index IF NOT EXISTS FOR (n:Chunk) ON (n.time);

// Install APOC library (if needed) - This is done via Docker environment variables instead
// Set up fulltext search for entity names
// Note: Fulltext indexes are typically created after data import in Neo4j 5.x
// For now we'll create a basic lookup structure

// Create a sample constraint to ensure uniqueness
CREATE CONSTRAINT entity_unique_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;

// Enable vector operations (if GDS is available)
// Note: These settings are typically configured in neo4j.conf
// This script just ensures basic schema elements exist

// Create a sample graph to verify the setup works
CREATE (n:GraphNode:Test {name: 'InitializationNode', created_at: datetime()}) RETURN n;