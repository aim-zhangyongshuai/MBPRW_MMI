# from .node_to_node import NodeToNode
#
# class ProteinToProtein(NodeToNode):
# 	def __init__(self, file_path, sep = "\t"):
# 		super().__init__(file_path, sep)
import pandas as pd
import networkx as nx

class ProteinToProtein:
    def __init__(self, directed, file_path, sep=","):  # 逗号分隔符
        self.file_path = file_path
        self.directed = directed
        self.sep = sep
        self.load()

    def load(self):
        self.load_df()
        self.load_edge_list()
        self.load_graph()

    def load_df(self):
        df = pd.read_csv(self.file_path, sep=self.sep)
        self.df = df

    def load_edge_list(self):
        assert self.df is not None
        edge_list = list(zip(self.df["HGNC_Symbol.1"], self.df["HGNC_Symbol.2"]))
        self.edge_list = edge_list

    def load_graph(self):
        self.graph = nx.DiGraph() if self.directed else nx.Graph()
        self.graph.add_edges_from(self.edge_list)
