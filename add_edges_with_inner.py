import numpy as np
import networkx as nx
import random

class inner_Add:
    def __init__(self, S, sub_graph, reconstructed_G):
        self.S = S
        self.g = sub_graph
        self.nodes = list(self.g.nodes())
        self.constructed_G = reconstructed_G

        self.disconnected_pairs = list(nx.non_edges(self.g))
        self.non_edges_similarty = {}

        for i in self.disconnected_pairs:
            self.non_edges_similarty[i] = self.S[[self.nodes.index(i[0])],[self.nodes.index(i[1])]]

        self.sorted_node_pairs = [key for key, value in sorted(self.non_edges_similarty.items(), key=lambda x: x[1], reverse=True)]

    def add(self, γ, η):
        edge_number = self.g.number_of_edges()

        first_add_edges_number = int(edge_number * γ)
        first_add_edges = []

        first_add_edges = self.sorted_node_pairs[:first_add_edges_number]

        last_add_edges_number = int(edge_number * η)

        random.shuffle(first_add_edges)
        
        self.constructed_G.add_edges_from(first_add_edges[:last_add_edges_number])

        return self.constructed_G


