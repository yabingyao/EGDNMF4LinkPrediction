import numpy as np
import networkx as nx
import random
import copy
from divide_network_based_random import Divide
from FSSDNMF_2layer import fssdnmf_2layer
from FSSDNMF import fssdnmf
from DNMF import dnmf
from NMF_algorithm import nmf
from add_edges_with_inner import inner_Add 
from add_edges_with_intro import intro_Add
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

class dnmfa_f(object):
    def __init__(self, train_graph):
        self.G = train_graph
        self.R = 20
        self.γ = 0.6
        self.η = 0.1

    def training(self):
        reconstructed_Gs = [copy.deepcopy(self.G)] * self.R

        divide_model = Divide(self.G)
        A_1, A_2, A_b, G1, G2, B = divide_model.divide_community()
        S_all = []

        S1_model = fssdnmf(A_1)
        S2_model = fssdnmf(A_2)
        Sb_model = fssdnmf(A_b)

        S1_model.pre_training()
        S1 = S1_model.training()

        S2_model.pre_training()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        S2 = S2_model.training()

        Sb_model.pre_training()
        Sb = Sb_model.training()

        for reconstructed_G in reconstructed_Gs:

            G1_add_model = inner_Add(S1, G1, reconstructed_G)
            reconstructed_G = G1_add_model.add(self.γ, self.η)

            G2_add_model = inner_Add(S2, G2, reconstructed_G)
            reconstructed_G = G2_add_model.add(self.γ, self.η)

            Gb_add_model = intro_Add(Sb, B, G1, G2, reconstructed_G)
            reconstructed_G = Gb_add_model.add(self.γ, self.η)

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


    def nmf(self, matrix):
        nmf_model = NMF(n_components=25,
                   init="random",
                   random_state=42,
                   max_iter=300)
        W = nmf_model.fit_transform(csr_matrix(matrix))
        H = nmf_model.components_
        S = np.dot(W, H)
        return S
