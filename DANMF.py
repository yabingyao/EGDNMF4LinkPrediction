import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

class danmf(object):
    def __init__(self, graph):
        self.graph = graph
        self.A = nx.adjacency_matrix(self.graph).toarray()
        self.L = nx.laplacian_matrix(self.graph).toarray()
        self.D = self.L+self.A
        self.lamb = 0.1
        #self.layers = [128, 64, 32]
        self.layers = [64, 32, 16]
        #self.layers = [32, 16, 8]
        self.p = len(self.layers)
        self.iterations = 100


    def setup_z(self, i):
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]

    def sklearn_pretrain(self, i):
        nmf_model = NMF(n_components=self.layers[i],
                        init="random",
                        random_state=42,
                        max_iter=100)

        U = nmf_model.fit_transform(self.Z)
        V = nmf_model.components_
        return U, V


    def pre_training(self):
        self.U_s = []
        self.V_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            U, V = self.sklearn_pretrain(i)
            self.U_s.append(U)
            self.V_s.append(V)


    def setup_Q(self):
        self.Q_s = [None for _ in range(self.p+1)]
        self.Q_s[self.p] = np.eye(self.layers[self.p-1])
        for i in range(self.p-1, -1, -1):
            self.Q_s[i] = np.dot(self.U_s[i], self.Q_s[i+1])

    def update_U(self, i):
        if i == 0:
            R = np.dot(self.U_s[0], self.Q_s[1]).dot(self.VpVpT).dot(self.Q_s[1].T)
            R = R + np.dot(self.A_sq, self.U_s[0]).dot(self.Q_s[1]).dot(self.Q_s[1].T)
            Ru = 2 * np.dot(self.A, self.V_s[self.p-1].T).dot(self.Q_s[1].T)
            self.U_s[0] = (self.U_s[0] * Ru) / np.maximum(R, 10**-10)
        else:
            R = np.dot(self.P.T, self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.VpVpT).dot(self.Q_s[i+1].T)
            R = R + np.dot(self.P.T, self.A_sq).dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.Q_s[i+1].T) 
            Ru = 2 * np.dot(self.P.T, self.A).dot(self.V_s[self.p-1].T).dot( self.Q_s[i+1].T)   
            self.U_s[i] = (self.U_s[i] * Ru) / np.maximum(R, 10**-10)


    def update_P(self, i):
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.dot(self.U_s[i])


    def update_V(self, i):
        if i < self.p-1:
            Vu = 2 * np.dot(self.P.T, self.A)
            Vd = np.dot(self.P.T, self.P).dot(self.V_s[i]) + self.V_s[i]
            self.V_s[i] = self.V_s[i] * (Vu / np.maximum(Vd, 10**-10))
        else:
            Vu = 2 * np.dot(self.P.T, self.A) + self.lamb * np.dot(self.V_s[i], self.A)
            Vd = np.dot(self.P.T, self.P).dot(self.V_s[i]) 
            Vd = Vd + self.V_s[i] + self.lamb * np.dot(self.V_s[i], self.D)
            self.V_s[i] = self.V_s[i] * (Vu / np.maximum(Vd, 10**-10))


    def training(self):
        self.A_sq = np.dot(self.A, self.A.T)
        for iteration in tqdm(range(self.iterations), desc="Training pass: ", leave=True):
            self.setup_Q()
            self.VpVpT = np.dot(self.V_s[self.p-1], self.V_s[self.p-1].T)
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
                self.update_V(i) 

        S = self.U_s[0].dot(self.U_s[1]).dot(self.V_s[1]) 
       
        return S

        