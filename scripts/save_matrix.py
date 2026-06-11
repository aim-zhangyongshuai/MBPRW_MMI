import numpy as np
import networkx as nx
import sys
import pickle
from multiprocessing import Pool
import pandas as pd
import os
from MMI_utils.MMI_remodify import MMI


def compute_shortest_paths(args):
    """
    Compute shortest paths from a single source node to all other nodes using Dijkstra's algorithm.
    Args:
        args: Tuple containing (networkx graph, source node, weight attribute name)
    Returns:
        Dictionary of target nodes and their corresponding shortest path lengths
    """
    network, node, weight = args
    lengths = nx.single_source_dijkstra_path_length(network, node, weight=weight)
    # Exclude self-loop and return path lengths to other nodes
    return {node2: length for node2, length in lengths.items() if node2 != node}


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    print(f"Project root: {project_root}")

    # List of lambda values to iterate over for edge weight configuration
    lambda_list = [0.1, 0.3, 0.4, 0.5, 0.7, 1.0, 1.0, 1.2, 1.7, 5]

    for n1 in lambda_list:
        # Define edge weights for the multi-modal interaction network
        weights = {
            "protein_protein": 1,
            "protein_biological_function": n1,
            "biological_function_biological_function": n1
        }

        # Initialize and load the MMI network with current weights
        mmi_network = MMI(weights=weights)
        mmi_network.load()

        output_dir = os.path.join(project_root, "output", "matrices", f"lambda_{n1}")
        # Create directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Save the complete network graph
        mmi_network.save_graph(output_dir)

        # Load the saved graph for further processing
        graph_path = os.path.join(output_dir, "graph.pkl")
        with open(graph_path, 'rb') as file:
            G = pickle.load(file)

        # Extract the largest connected component (LCC) of the graph
        components = nx.connected_components(G)
        lcc_list = sorted(list(components), key=len, reverse=True)
        G1 = G.subgraph(lcc_list[0]).copy()

        # Filter nodes: retain only protein nodes (exclude GO terms)
        protein_nodes = [node for node in G1.nodes() if not node.startswith("GO:")]
        protein_nodes = sorted(protein_nodes)
        nodes_in_subgraph = protein_nodes

        # Create mapping from node name to matrix index
        node_to_index = {node: idx for idx, node in enumerate(nodes_in_subgraph)}
        num_nodes = len(nodes_in_subgraph)

        # Initialize distance matrix with infinity (unreachable pairs)
        adj_matrix = np.full((num_nodes, num_nodes), np.inf)
        # Set diagonal to 0 (distance from node to itself)
        np.fill_diagonal(adj_matrix, 0)

        # Compute shortest paths in parallel for all protein nodes
        with Pool() as pool:
            path_results = pool.map(
                compute_shortest_paths,
                [(G1, node, 'weight') for node in nodes_in_subgraph]
            )

        # Fill the adjacency matrix with computed shortest path lengths
        for source_node, connections in zip(nodes_in_subgraph, path_results):
            i = node_to_index[source_node]
            for target_node, length in connections.items():
                if target_node in node_to_index:
                    j = node_to_index[target_node]
                    adj_matrix[i][j] = length
                    adj_matrix[j][i] = length  # Symmetric for undirected graph

        # Save node index mapping and distance matrix
        index_save_path = os.path.join(output_dir, f"node_to_index_1_{n1}_{n1}.pkl")
        matrix_save_path = os.path.join(output_dir, f"adj_matrix_1_{n1}_{n1}.pkl")

        with open(index_save_path, 'wb') as f:
            pickle.dump(node_to_index, f)
        with open(matrix_save_path, 'wb') as f:
            pickle.dump(adj_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Matrix for lambda = {n1} saved successfully to {output_dir}")