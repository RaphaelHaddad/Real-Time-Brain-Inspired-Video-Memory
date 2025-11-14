import json
import uuid
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from ..core.logger import get_logger
from ..components.neo4j_handler import Neo4jHandler

logger = get_logger(__name__)


def serialize_for_json(obj: Any) -> Any:
    """
    Convert Neo4j objects to JSON-serializable formats.
    Handles DateTime, Point, and other Neo4j-specific types.
    """
    if hasattr(obj, 'isoformat'):  # DateTime, Date, Time objects
        return obj.isoformat()
    elif hasattr(obj, 'x') and hasattr(obj, 'y'):  # Point objects
        return {"type": "Point", "x": obj.x, "y": obj.y, "srid": getattr(obj, 'srid', None)}
    elif hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'z'):  # 3D Point objects
        return {"type": "Point3D", "x": obj.x, "y": obj.y, "z": obj.z, "srid": getattr(obj, 'srid', None)}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    else:
        # Try to convert to string for unknown types
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


class GraphExporter:
    """Handles exporting and importing knowledge graphs for collaboration"""
    
    def __init__(self, neo4j_handler: Neo4jHandler):
        self.neo4j_handler = neo4j_handler
    
    async def export_graph(self, graph_uuid: str, output_path: str) -> str:
        """
        Export a knowledge graph with the specified UUID to a JSON file
        for collaboration and transfer to other Neo4j instances.
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Exporting graph with UUID: {graph_uuid}")
            
            # Export nodes and relationships for the specific graph
            nodes_data = await self._export_nodes(graph_uuid)
            relationships_data = await self._export_relationships(graph_uuid)
            
            export_data = {
                "graph_uuid": graph_uuid,
                "export_timestamp": str(uuid.uuid4()),  # Could use actual timestamp
                "nodes": nodes_data,
                "relationships": relationships_data,
                "export_format_version": "1.0"
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph exported successfully to: {output_file}")
            logger.info(f"Exported {len(nodes_data)} nodes and {len(relationships_data)} relationships")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting graph: {str(e)}")
            raise
    
    async def _export_nodes(self, graph_uuid: str) -> List[Dict[str, Any]]:
        """Export all nodes for the specified graph UUID"""
        async with self.neo4j_handler.driver.session() as session:
            result = await session.run(
                """
                MATCH (n:GraphNode)
                WHERE n.graph_uuid = $graph_uuid
                RETURN n.name as name, labels(n) as labels, properties(n) as properties
                """,
                graph_uuid=graph_uuid
            )
            
            nodes = []
            async for record in result:
                # Remove graph_uuid from properties to avoid duplication
                props = dict(record["properties"])
                props.pop("graph_uuid", None)
                
                # Serialize properties to handle DateTime and other Neo4j objects
                serialized_props = serialize_for_json(props)
                
                nodes.append({
                    "name": record["name"],
                    "labels": [label for label in record["labels"] if label != "GraphNode"],
                    "properties": serialized_props
                })
            
            return nodes
    
    async def _export_relationships(self, graph_uuid: str) -> List[Dict[str, Any]]:
        """Export all relationships for the specified graph UUID"""
        async with self.neo4j_handler.driver.session() as session:
            result = await session.run(
                """
                MATCH (a:GraphNode)-[r]->(b:GraphNode)
                WHERE a.graph_uuid = $graph_uuid AND b.graph_uuid = $graph_uuid
                RETURN type(r) as type, a.name as from_node, b.name as to_node, properties(r) as properties
                """,
                graph_uuid=graph_uuid
            )
            
            relationships = []
            async for record in result:
                # Remove graph_uuid from properties to avoid duplication
                props = dict(record["properties"])
                props.pop("graph_uuid", None)
                
                # Serialize properties to handle DateTime and other Neo4j objects
                serialized_props = serialize_for_json(props)
                
                relationships.append({
                    "type": record["type"],
                    "from_node": record["from_node"],
                    "to_node": record["to_node"],
                    "properties": serialized_props
                })
            
            return relationships


class GraphImporter:
    """Handles importing knowledge graphs from exported files"""
    
    def __init__(self, neo4j_handler: Neo4jHandler):
        self.neo4j_handler = neo4j_handler
    
    async def import_graph(self, input_path: str, new_uuid: str = None) -> str:
        """
        Import a knowledge graph from an exported JSON file.
        If new_uuid is provided, it will replace the original UUID in the imported graph.
        """
        try:
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Import file does not exist: {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Use provided UUID or the one from the export file
            target_uuid = new_uuid or import_data.get("graph_uuid", str(uuid.uuid4()))
            
            logger.info(f"Importing graph with UUID: {target_uuid}")
            
            # Import nodes first
            await self._import_nodes(import_data["nodes"], target_uuid)
            
            # Then import relationships
            await self._import_relationships(import_data["relationships"], target_uuid)
            
            logger.info(f"Graph imported successfully with UUID: {target_uuid}")
            logger.info(f"Imported {len(import_data['nodes'])} nodes and {len(import_data['relationships'])} relationships")
            
            return target_uuid
            
        except Exception as e:
            logger.error(f"Error importing graph: {str(e)}")
            raise
    
    async def _import_nodes(self, nodes: List[Dict], graph_uuid: str):
        """Import nodes into the Neo4j database"""
        async with self.neo4j_handler.driver.session() as session:
            for node in nodes:
                # Skip nodes with null or missing names as they cannot be merged
                if not node.get("name"):
                    logger.warning(f"Skipping node with missing or null name: {node}")
                    continue
                
                # Build dynamic labels
                label_str = ":".join(node["labels"]) if node["labels"] else "Entity"
                
                # Add graph_uuid to properties
                props = {**node["properties"], "graph_uuid": graph_uuid, "name": node["name"]}
                
                # Build the property string dynamically
                prop_string = ", ".join([f"{k}: ${k}" for k in props.keys()])
                
                query = f"""
                MERGE (n:{label_str} {{name: $name, graph_uuid: $graph_uuid}})
                SET n += $properties
                """
                
                await session.run(
                    query,
                    name=node["name"],
                    graph_uuid=graph_uuid,
                    properties=props
                )
    
    async def _import_relationships(self, relationships: List[Dict], graph_uuid: str):
        """Import relationships into the Neo4j database"""
        async with self.neo4j_handler.driver.session() as session:
            for rel in relationships:
                # Skip relationships with null node names
                if rel["from_node"] is None or rel["to_node"] is None:
                    logger.warning(f"Skipping relationship with null node names: {rel}")
                    continue
                
                query = f"""
                MATCH (a:GraphNode {{name: $from_node, graph_uuid: $graph_uuid}})
                MATCH (b:GraphNode {{name: $to_node, graph_uuid: $graph_uuid}})
                MERGE (a)-[r:`{rel['type']}` {{graph_uuid: $graph_uuid}}]->(b)
                """
                query = f"""
                MATCH (a:GraphNode {{name: $from_node, graph_uuid: $graph_uuid}})
                MATCH (b:GraphNode {{name: $to_node, graph_uuid: $graph_uuid}})
                MERGE (a)-[r:`{rel['type']}` {{graph_uuid: $graph_uuid}}]->(b)
                """
                
                await session.run(
                    query,
                    from_node=rel["from_node"],
                    to_node=rel["to_node"],
                    graph_uuid=graph_uuid
                )