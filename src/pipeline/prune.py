import json
import copy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

class Graph:
    def __init__(self, graph_data):
        self.nodes = graph_data['nodes']
        self.node_name_list = self._get_node_name_list()
        self.relationships = graph_data['relationships']
        self.original_node_count = len(self.nodes)
        self.original_rel_count = len(self.relationships)
        self.relelation_types = self._get_relation_types()
        self.embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
    def _get_node_name_list(self):
        name_list = []
        for node in self.nodes:
            name_list.append(node['name'])
        return name_list
    def _get_relation_types(self):
        types = []
        for relation in self.relationships:
            types.append(relation['type'])
        return types

    def nodes_with_same_edges(self, relation):
        result = []
        for rel in self.relationships:
            if rel.get("type") == relation:
                result.append({
                    'from_node': rel.get('from_node'),
                    'to_node': rel.get('to_node')
                })
        return result
    
    def relation_between_two_nodes(self, node1, node2):
        "return all relationship between two given nodes"
        results = []

        for rel in self.relationships:
            fn = rel.get("from_node")
            tn = rel.get("to_node")

            # Check both directions
            if (fn == node1 and tn == node2) or (fn == node2 and tn == node1):
                results.append({
                    "type": rel.get("type"),
                    "from_node": fn,
                    "to_node": tn,
                    "properties": rel.get("properties")
                })

        return results
            
    def _get_representative_relation(self, relation_sentences):
        '''
        Return index of the most representative relation among many directed relation closest to central meaning.
        '''
        embeddings = self.embedding_model.encode(relation_sentences)
        centroid = np.mean(embeddings, axis=0)
        similarities = cosine_similarity([centroid], embeddings)[0]
        print("\nSIMILARITIES: \n", similarities)
        most_representative_idx = np.argmax(similarities)
        return most_representative_idx

    def _are_same_context(self, relation_sentences, threshold=0.8):
        """
        Determine representative relation and outliers.
        Steps:
        1. Compute similarity matrix.
        2. Build graph: edge exists if similarity >= threshold.
        3. Each connected component = cluster of similar relations.
        4. Largest cluster = main meaning.
        5. Choose the representative from the largest cluster.
        
        Returns:
            should_prune (bool)
            representative_idx (int)
            outlier_indices (list[int])
        """
        if len(relation_sentences) <= 1:
                return False, None, list(range(len(relation_sentences)))

        # 1. Encode embeddings
        embeddings = self.embedding_model.encode(relation_sentences)
        similarity_matrix = cosine_similarity(embeddings)

        # 2. Build graph based on similarity threshold
        G = nx.Graph()
        G.add_nodes_from(range(len(relation_sentences)))

        for i in range(len(relation_sentences)):
            for j in range(i + 1, len(relation_sentences)):
                if similarity_matrix[i][j] >= threshold:
                    G.add_edge(i, j)

        # 3. Extract clusters (connected components)
        clusters = [list(c) for c in nx.connected_components(G)]
        
        # If every sentence is isolated → no pruning (each is an outlier)
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

        should_prune = len(main_cluster) > 1  # prune only if at least 2 similar

        return should_prune, representative_idx, outlier_indices

    def prune_graph(self, similarity_threshold=0.8):
        processed_pairs = set()
        new_relationships = []
        
        for rel in self.relationships:
            node1 = rel['from_node']
            node2 = rel['to_node']
            
            # pair key (sorted to treat A-B and B-A as same pair)
            pair_key = tuple(sorted([node1, node2]))
            
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            relations = self.relation_between_two_nodes(node1, node2)
            
            if len(relations) <= 1:
                new_relationships.extend(relations)
                continue
            
            node1_to_node2 = [r for r in relations if r['from_node'] == node1 and r['to_node'] == node2]
            node2_to_node1 = [r for r in relations if r['from_node'] == node2 and r['to_node'] == node1]
            
            for direction_relations in [node1_to_node2, node2_to_node1]:
                if not direction_relations:
                    continue
                
                if len(direction_relations) == 1:
                    new_relationships.append(direction_relations[0])
                else:
                    relation_sentences = []
                    for r in direction_relations:
                        rel_type = r['type'].replace("_", " ")
                        sentence = f"{r['from_node']} {rel_type} {r['to_node']}"
                        relation_sentences.append(sentence)
                    should_prune, representative_idx, outlier_indices = self._are_same_context(
                        relation_sentences, similarity_threshold
                    )
                    print("\nSHOULD PRUNE: \n", should_prune )
                    
                    if should_prune:
                        # Keep the most representative relation from similar cluster
                        new_relationships.append(direction_relations[representative_idx])
                        
                        # Keep all outlier relations
                        for outlier_idx in outlier_indices:
                            new_relationships.append(direction_relations[outlier_idx])
                        
                        kept_count = 1 + len(outlier_indices)
                        print(f"Pruned {len(direction_relations)} relations to {kept_count} between {direction_relations[0]['from_node']} → {direction_relations[0]['to_node']} (kept {len(outlier_indices)} outliers)")
                    else:
                        new_relationships.extend(direction_relations)
        

        return new_relationships
        

def main():
    input_file = "data/export_graph/exported_graph_mvp_8679eaa5-1b21-4975-b7d6-c86e3a9c958d.json"
    output_file = "data/outputs/pruned_with_embedding_graph_8679eaa5-1b21-4975-b7d6-c86e3a9c958d.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    graph = Graph(graph_data)
    relation_type_list = graph.relelation_types
    print("Node Count: ", graph.original_node_count)
    print("Relation Count (edges): ", graph.original_rel_count)
    print("Relation Name List length: \n", len(relation_type_list))
  
    same_eges = graph.nodes_with_same_edges(relation_type_list[0])
    print(f"Nodes with {relation_type_list[0]} relation: \n", same_eges)
    node1, node2, = graph.nodes[0]['name'], graph.nodes[1]['name']
    node1, node2, = "Liquid", "Flask"
    relation_between_two_nodes = graph.relation_between_two_nodes(node1, node2)
    print(f"Relations between {node1} and {node2}: \n", relation_between_two_nodes)
    print("\n PRUNING graph \n")
    
    new_relationships = graph.prune_graph()
    print("New relation length: ", len(new_relationships))
    pruned_graph = copy.deepcopy(graph_data)
    pruned_graph['relationships'] = new_relationships
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pruned_graph, f, indent=2)
    print(f"Pruned graph saved to {output_file}")
    
    pruned_graph_info = Graph(pruned_graph)
    print("After Pruning: \n")
    print("Pruned Node count: ", pruned_graph_info.original_node_count)
    print("Pruned Relationship COunt: ", pruned_graph_info.original_rel_count)
    print("Relation Name List length: \n", len(pruned_graph_info.relelation_types))
    relation_between_two_nodes = pruned_graph_info.relation_between_two_nodes(node1, node2)
    print(f"Relations between {node1} and {node2} After Pruning: \n", relation_between_two_nodes)

if __name__ == "__main__":
    main()