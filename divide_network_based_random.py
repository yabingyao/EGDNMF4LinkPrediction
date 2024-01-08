import numpy as np
import networkx as nx

class Divide:
    def __init__(self, g):
        self.G = g


    def divide_community(self):
        nodes = list(self.G.nodes())
        half_nodes_number = int(len(nodes) / 2)

        communities = [nodes[:half_nodes_number], nodes[half_nodes_number:]]

        G1 = self.G.subgraph(communities[0])
        G2 = self.G.subgraph(communities[1])

        A_1 = nx.adjacency_matrix(G1)
        A_2 = nx.adjacency_matrix(G2)

        G1_nodes = list(G1.nodes())
        G2_nodes= list(G2.nodes())

        # 创建一个一般的网络
        B = nx.Graph()
        B.add_nodes_from(G1_nodes, bipartite=0) # 添加一类节点
        B.add_nodes_from(G2_nodes, bipartite=1) # 添加另一类节点
        for i in G1_nodes:
            for j in G2_nodes:
                if self.G.has_edge(i, j):
                    B.add_edges_from([(i, j)]) # 添加边

        A_b = nx.bipartite.biadjacency_matrix(B, row_order=G1_nodes, column_order=G2_nodes) # A_b.toarray()

        return A_1, A_2, A_b, G1, G2, B