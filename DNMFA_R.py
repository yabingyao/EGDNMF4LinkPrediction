import numpy as np
import networkx as nx
import random
import copy
from divide_network_based_random import Divide
from DNMF import dnmf
from NMF_algorithm import nmf
from add_edges_with_inner import inner_Add 
from add_edges_with_intro import intro_Add
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

class dnmfa_r(object):
    def __init__(self, train_graph):
        self.G = train_graph
        self.R = 20
        self.η = 0.1

    def training(self):
        reconstructed_Gs = [copy.deepcopy(self.G)] * self.R

        divide_model = Divide(self.G)
        A_1, A_2, A_b, G1, G2, B = divide_model.divide_community()
        S_all = []

        for reconstructed_G in reconstructed_Gs:

            reconstructed_G = self.indd(G1, reconstructed_G)

            reconstructed_G = self.indd(G2, reconstructed_G)

            reconstructed_G = self.itadd(B, G1, G2, reconstructed_G)

            reconstructed_A = nx.adjacency_matrix(reconstructed_G)
            #Similar = self.nmf(reconstructed_A) 

            Similar_model = nmf(reconstructed_A)
            Similar = Similar_model.training()  
            S_all.append(Similar)

        sum_matrix = S_all[0]
        for matrix in S_all[1:]:
            sum_matrix += matrix
        S = 1/self.R * sum_matrix
        
        return S



    def indd(self, sub_G, reconstructed_G):
        added_number = int(round(self.η * len(sub_G.edges())))
        unconnected_edges = list(nx.non_edges(sub_G))
        random.shuffle(unconnected_edges)
        added_edges = unconnected_edges[:added_number]
        reconstructed_G.add_edges_from(added_edges)

        return reconstructed_G



    def itadd(self, B, G1, G2, reconstructed_G):
        added_number = int(round(self.η * len(B.edges())))

        G1_nodes = list(G1.nodes())
        G2_nodes = list(G2.nodes())

        unconnected_edges = []
        for i in G1_nodes:
            for j in G2_nodes:
                if not B.has_edge(i, j):
                    unconnected_edges.append((i, j))

        random.shuffle(unconnected_edges)
        added_edges = unconnected_edges[:added_number]
        reconstructed_G.add_edges_from(added_edges)
        
        return reconstructed_G


    def nmf(self, matrix):
        nmf_model = NMF(n_components=25,
                   init="random",
                   random_state=42,
                   max_iter=300)
        W = nmf_model.fit_transform(csr_matrix(matrix))
        H = nmf_model.components_
        S = np.dot(W, H)
        return S
