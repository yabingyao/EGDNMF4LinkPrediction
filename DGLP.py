import numpy as np
import networkx as nx

class dglp(object):
    def __init__(self, train_graph):
        self.G = train_graph


    def training(self, sorce_node, target_node):
        path_bool =  nx.has_path(self.G, source=sorce_node, target=target_node)
        if path_bool:
            shortest_path_length = nx.shortest_path_length(self.G, source=sorce_node, target=target_node)
            degree_sorce_node = self.G.degree(sorce_node)
            degree_target_node = self.G.degree(target_node)

            common_neighbors = list(nx.common_neighbors(self.G, sorce_node, target_node))

            neighbors_degrees = 0
            for node in common_neighbors:
                neighbors_degrees += self.G.degree(node)

            S = (degree_sorce_node + degree_target_node) / (shortest_path_length + 1) + neighbors_degrees
        else:
            common_neighbors = list(nx.common_neighbors(self.G, sorce_node, target_node))
            neighbors_degrees = 0
            for node in common_neighbors:
                neighbors_degrees += self.G.degree(node)

            S = neighbors_degrees

        return S