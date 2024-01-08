import numpy as np
import networkx as nx
#from networkx.algorithms import bipartite
import random

class intro_Add:
    def __init__(self, S, B, G1, G2, reconstructed_G):
        self.S = S
        self.b = B
        self.G1_nodes = list(G1.nodes())
        self.G2_nodes = list(G2.nodes())
        self.constructed_G = reconstructed_G

        self.disconnected_pairs = []
        for i in self.G1_nodes:
            for j in self.G2_nodes:
                if not self.b.has_edge(i, j):
                    self.disconnected_pairs.append((i, j))

        self.non_edges_similarty = {}
        for i in self.disconnected_pairs:
            self.non_edges_similarty[i] = self.S[[self.G1_nodes.index(i[0])],[self.G2_nodes.index(i[1])]]

        self.sorted_node_pairs = [key for key, value in sorted(self.non_edges_similarty.items(), key=lambda x: x[1], reverse=True)]

    def add(self, γ, η):
        edge_number = self.b.number_of_edges()
        
        first_add_edges_number = int(edge_number * γ)
        first_add_edges = []

        first_add_edges = self.sorted_node_pairs[:first_add_edges_number]

        last_add_edges_number =  int(edge_number * η)

        random.shuffle(first_add_edges)
        self.constructed_G.add_edges_from(first_add_edges[:last_add_edges_number])

        return self.constructed_G