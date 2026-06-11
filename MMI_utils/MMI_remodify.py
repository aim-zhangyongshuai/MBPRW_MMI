import numpy
import numpy as np
import networkx as nx
import os
import pickle
import pandas as pd
from utils.guney_code import network_utilities

from utils.guney_code.wrappers import get_random_nodes
from .protein_to_protein import ProteinToProtein
from .protein_to_biological_function import ProteinToBiologicalFunction
from .biological_function_to_biological_function import BiologicalFunctionToBiologicalFunction


script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
# ==================================================================

PROTEIN = "protein"
BIOLOGICAL_FUNCTION = "biological_function"
WEIGHT = "weight"
PROTEIN_PROTEIN = "protein-protein"
PROTEIN_BIOLOGICAL_FUNCTION = "protein-biological_function"
BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION = "biological_function-biological_function"


class MMI():
    def __init__(self, nodes=[PROTEIN, BIOLOGICAL_FUNCTION],
                 edges=[PROTEIN_PROTEIN, PROTEIN_BIOLOGICAL_FUNCTION, BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION],
                 weights=None,
                 protein2protein_file_path=os.path.join(PROJECT_ROOT, "data", "network_edge", "ncPPI_PPI_filter_RNA_lcc.csv"),
                 protein2protein_directed=False,
                 protein2biological_function_file_path=os.path.join(PROJECT_ROOT, "data", "network_edge", "Protein_to_GO_formal_version.csv"),
                 protein2biological_function_directed=False,
                 biological_function2biological_function_file_path=os.path.join(PROJECT_ROOT, "data", "network_edge", "GO_to_GO_filtered.csv"),
                 biological_function2biological_function_directed=False):
        # Parameters
        self.nodes = nodes
        self.edges = edges
        self.weights = weights

        # File paths
        self.protein2protein_file_path = protein2protein_file_path
        self.protein2biological_function_file_path = protein2biological_function_file_path
        self.biological_function2biological_function_file_path = biological_function2biological_function_file_path

        # Directed
        self.protein2protein_directed = protein2protein_directed
        self.protein2biological_function_directed = protein2biological_function_directed
        self.biological_function2biological_function_directed = biological_function2biological_function_directed

    def load_graph(self):
        # Initialize graph
        self.graph = nx.Graph()

        # Load components and add edges as appropriate
        self.components = dict()

        if (PROTEIN in self.nodes) and (PROTEIN_PROTEIN in self.edges):
            self.components["protein_to_protein"] = ProteinToProtein(self.protein2protein_directed,
                                                                     self.protein2protein_file_path)
            self.add_edges(self.components["protein_to_protein"].edge_list, PROTEIN, PROTEIN)

        if (BIOLOGICAL_FUNCTION in self.nodes) and (PROTEIN_BIOLOGICAL_FUNCTION in self.edges):
            self.components["protein_to_biological_function"] = ProteinToBiologicalFunction(
                self.protein2biological_function_directed, self.protein2biological_function_file_path)

            self.add_edges(self.components["protein_to_biological_function"].edge_list, PROTEIN, BIOLOGICAL_FUNCTION)

        if (BIOLOGICAL_FUNCTION in self.nodes) and (BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION in self.edges):
            self.components["biological_function_to_biological_function"] = BiologicalFunctionToBiologicalFunction(
                self.biological_function2biological_function_directed,
                self.biological_function2biological_function_file_path)
            self.add_edges(self.components["biological_function_to_biological_function"].edge_list, BIOLOGICAL_FUNCTION,
                           BIOLOGICAL_FUNCTION)

    def add_edges(self, edge_list, from_node_type, to_node_type):
        for from_node, to_node in edge_list:
            self.graph.add_edge(from_node, to_node)
            self.graph.nodes[from_node]["type"] = from_node_type
            self.graph.nodes[to_node]["type"] = to_node_type

    def weight_graph(self):
        self.create_class_specific_adjacency_dictionary()
        for from_node, adj_dict in self.cs_adj_dict.items():
            from_node_type = self.graph.nodes[from_node]["type"]
            for node_type, to_nodes in adj_dict.items():
                num_typed_nodes = len(to_nodes)
                for to_node in to_nodes:
                    if from_node_type == PROTEIN and node_type == PROTEIN:
                        edge_weight = self.weights.get("protein_protein", 1.0)
                    elif (from_node_type == PROTEIN and node_type == BIOLOGICAL_FUNCTION) or (
                            from_node_type == BIOLOGICAL_FUNCTION and node_type == PROTEIN):
                        edge_weight = self.weights.get("protein_biological_function", 1.0)
                    elif from_node_type == BIOLOGICAL_FUNCTION and node_type == BIOLOGICAL_FUNCTION:
                        edge_weight = self.weights.get("biological_function_biological_function", 1.0)
                    else:
                        continue

                    self.graph[from_node][to_node][WEIGHT] = edge_weight

    def create_class_specific_adjacency_dictionary(self):
        self.cs_adj_dict = {node: {} for node in self.graph.nodes()}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]["type"]
            neighbors = self.graph.neighbors(node)
            for neighbor in neighbors:
                neighbor_type = self.graph.nodes[neighbor]["type"]
                self.add_to_cs_adj_dict(node, neighbor_type, neighbor)

    def add_to_cs_adj_dict(self, node, successor_type, successor):
        if (successor_type in self.cs_adj_dict[node]):
            self.cs_adj_dict[node][successor_type].append(successor)
        else:
            self.cs_adj_dict[node][successor_type] = [successor]

    def load(self):
        self.load_graph()
        self.weight_graph()

    def save_graph(self, save_load_file_path):
        graph_file_path = os.path.join(save_load_file_path, "graph.pkl")
        with open(graph_file_path, "wb") as f:
            pickle.dump(self.graph, f)

    def calculate_closest_distance(self, adj_matrix, node_to_index, nodes_from, nodes_to):
        distances = []
        for node1 in nodes_from:
            values = []
            i = node_to_index[node1]
            for node2 in nodes_to:
                j = node_to_index[node2]
                values.append(adj_matrix[i, j])
            distances.append(np.min(values))
        return np.mean(distances)

    def calculate_proximity_with_weights(self, adj_matrix, node_to_index, network, nodes_from, nodes_to, lengths=None,
                                         nodes_from_random=None,
                                         nodes_to_random=None,
                                         bins=None, n_random=1000, min_bin_size=100, seed=452456, weight='weight'):
        nodes_network = set(network.nodes())
        nodes_from = set(nodes_from) & nodes_network
        nodes_to = set(nodes_to) & nodes_network
        if len(nodes_from) == 0 or len(nodes_to) == 0:
            return None
        d = self.calculate_closest_distance(adj_matrix, node_to_index, nodes_from, nodes_to)
        if bins is None and (nodes_from_random is None or nodes_to_random is None):
            bins = network_utilities.get_degree_binning(network, min_bin_size, lengths)
        if nodes_from_random is None:
            nodes_from_random = get_random_nodes(nodes_from, network, bins=bins, n_random=n_random,
                                                 min_bin_size=min_bin_size, seed=seed)
        if nodes_to_random is None:
            nodes_to_random = get_random_nodes(nodes_to, network, bins=bins, n_random=n_random,
                                               min_bin_size=min_bin_size, seed=seed)
        random_values_list = zip(nodes_from_random, nodes_to_random)
        values = np.empty(len(nodes_from_random))
        for i, values_random in enumerate(random_values_list):
            nodes_from, nodes_to = values_random
            values[i] = self.calculate_closest_distance(adj_matrix, node_to_index, nodes_from, nodes_to)
        m, s = np.mean(values), np.std(values)
        z = (d - m) / s if s != 0 else 0.0
        return d, z, (m, s)

    def calculate_proximity_with_weights_ncRNA(self, adj_matrix, node_to_index, network1, network2, nodes_from,
                                               nodes_to, lengths=None, nodes_from_random=None,
                                               nodes_to_random=None,
                                               bins=None, n_random=1000, min_bin_size=100, seed=452456,
                                               weight='weight'):
        nodes_network1 = set(network1.nodes())
        nodes_network2 = set(network2.nodes())
        nodes_from = set(nodes_from) & nodes_network1
        nodes_to = set(nodes_to) & nodes_network2
        if len(nodes_from) == 0 or len(nodes_to) == 0:
            return None
        d = self.calculate_closest_distance(adj_matrix, node_to_index, nodes_from, nodes_to)
        if bins is None and (nodes_from_random is None or nodes_to_random is None):
            bins1 = network_utilities.get_degree_binning(network1, min_bin_size, lengths)
            bins2 = network_utilities.get_degree_binning(network2, min_bin_size, lengths)
        if nodes_from_random is None:
            nodes_from_random = get_random_nodes(nodes_from, network1, bins=bins1, n_random=n_random,
                                                 min_bin_size=min_bin_size, seed=seed)
        if nodes_to_random is None:
            nodes_to_random = get_random_nodes(nodes_to, network2, bins=bins2, n_random=n_random,
                                               min_bin_size=min_bin_size, seed=seed)
        random_values_list = zip(nodes_from_random, nodes_to_random)
        values = np.empty(len(nodes_from_random))
        for i, values_random in enumerate(random_values_list):
            nodes_from, nodes_to = values_random
            values[i] = self.calculate_closest_distance(adj_matrix, node_to_index, nodes_from, nodes_to)
        m, s = np.mean(values), np.std(values)
        z = (d - m) / s if s != 0 else 0.0
        return d, z, (m, s)