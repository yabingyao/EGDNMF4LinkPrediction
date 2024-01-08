import numpy as np
import networkx as nx
import random
import copy
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

class nmfa1(object):
    def __init__(self, graph):
        self.G = graph
        self.R = 20
        self.η = 0.1
        self.added_number = int(round(self.η * len(self.G.edges())))

    def training(self):
        self.S_matrix = []
        reconstructed_Gs = [copy.deepcopy(self.G)] * self.R

        unconnected_edges = list(nx.non_edges(self.G))

        for perturbed_G in reconstructed_Gs:

            random.shuffle(unconnected_edges)
            added_edges = unconnected_edges[:self.added_number]

            perturbed_G.add_edges_from(added_edges)
            perturbed_matrix = nx.adjacency_matrix(perturbed_G)

            S = self.nmf(perturbed_matrix)
            self.S_matrix.append(S)

        sum_matrix = self.S_matrix[0]
        for matrix in self.S_matrix[1:]:
            sum_matrix += matrix

        similar_matrix = 1/self.R * sum_matrix

        return similar_matrix    


    def nmf(self, matrix):
        nmf_model = NMF(n_components=25,
                   init="random",
                   random_state=42,
                   max_iter=300)
        W = nmf_model.fit_transform(csr_matrix(matrix))
        H = nmf_model.components_
        S = np.dot(W, H)
        return S
