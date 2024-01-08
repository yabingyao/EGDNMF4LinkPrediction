import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

class fssdnmf_2layer(object):
    def __init__(self, X):
        self.A = X.toarray()
        self.S = self.A.dot(self.A)
        self.alfa = 0.001
        self.beta = 0.4
        self.layers = [64, 32]
        self.p = len(self.layers)
        self.iterations = 100


    def setup_zh(self, i):
        if i == 0:
            self.Z = csr_matrix(self.A)
        else:
            self.Z = csr_matrix(self.H_s[i-1])


    def setup_zu(self, i):
        if i == 0:
            self.ZZ = csr_matrix(self.S)
        else:
            self.ZZ = csr_matrix(self.U_s[i-1])


    def sklearn_pretrain_h(self, i):
        nmf_model = NMF(n_components=self.layers[i],
                        init="random",
                        random_state=42,
                        max_iter=200)
        W = nmf_model.fit_transform(self.Z)
        H = nmf_model.components_
        return W, H


    def sklearn_pretrain_u(self, i):
        nmf_model = NMF(n_components=self.layers[i],
                        init="random",
                        random_state=42,
                        max_iter=200)
        S = nmf_model.fit_transform(self.ZZ)
        U = nmf_model.components_
        return S, U


    def pre_training(self):
        self.W_s = []
        self.H_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_zh(i)
            W, H = self.sklearn_pretrain_h(i)
            self.W_s.append(W)
            self.H_s.append(H)

        self.S_s = []
        self.U_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_zu(i)
            S, U = self.sklearn_pretrain_u(i)
            self.S_s.append(S)
            self.U_s.append(U)


    def update_W(self, i):
        if i == 0:
            Y = self.W_s[0].dot(self.W_s[1]).dot(self.H_s[1].dot(self.H_s[1].T).dot(self.W_s[1].T))
            Y = Y+self.alfa*self.W_s[0].dot(self.W_s[1]).dot(self.U_s[1]).dot(self.U_s[1].T).dot(self.W_s[1].T)
            Ru = self.A.dot(self.H_s[1].T.dot(self.W_s[1].T)) + self.alfa*self.S.dot(self.U_s[1].T).dot(self.W_s[1].T)
            self.W_s[0] = (self.W_s[0]*Ru)/np.maximum(Y, 10**-10)
        else:
            Y = self.W_s[0].T.dot(self.W_s[0]).dot(self.W_s[1]).dot(self.H_s[1]).dot(self.H_s[1].T)
            Y = Y+self.alfa*self.W_s[0].T.dot(self.W_s[0]).dot(self.W_s[1]).dot(self.U_s[1]).dot(self.U_s[1].T)
            Ru = self.W_s[0].T.dot(self.A).dot(self.H_s[1].T) + self.alfa*self.W_s[0].T.dot(self.S.dot(self.U_s[1].T)) 
            self.W_s[1] = (self.W_s[1]*Ru)/np.maximum(Y, 10**-10)


    def update_H(self, i):
        if i == 0:
            X = np.linalg.norm(self.H_s[0], axis=0)
            D1 = np.linalg.norm(X, ord=1)
            Hu = self.W_s[0].T.dot(self.A)
            Hd = self.W_s[0].T.dot(self.W_s[0]).dot(self.H_s[0])+self.beta*self.H_s[0]*D1
            self.H_s[0] = self.H_s[0] * Hu/np.maximum(Hd, 10**-10)
        else:
            X = np.linalg.norm(self.H_s[1], axis=0)
            D1 = np.linalg.norm(X, ord=1)
            Hu = self.W_s[1].T.dot(self.W_s[0].T).dot(self.A)
            Hd = self.W_s[1].T.dot(self.W_s[0].T).dot(self.W_s[0]).dot(self.W_s[1]).dot(self.H_s[1]) + self.beta*self.H_s[1]*D1
            self.H_s[1] = self.H_s[1] * Hu/np.maximum(Hd, 10**-10)


    def training(self):
        for iteration in tqdm(range(self.iterations), desc="Training pass: ", leave=True):
            for i in range(self.p):
                self.update_W(i)
                self.update_H(i)

        S = self.W_s[0].dot(self.W_s[1]).dot(self.H_s[1])

        return S
