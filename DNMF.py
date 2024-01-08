import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

class dnmf(object):
    def __init__(self, A):
        self.A = A.toarray()
        #self.layers = [128, 64, 32]
        self.layers = [64, 32, 16]
        #self.layers = [32, 16, 8]
        self.p = len(self.layers)

    def sklearn_pretrain(self, i):
        nmf_model = NMF(n_components=self.layers[i],
                        init="random",
                        random_state=42,
                        max_iter=100)

        W = nmf_model.fit_transform(csr_matrix(self.Z))
        H = nmf_model.components_
        return W, H

    def setup_z(self, i):
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.W_s[i-1]


    def pre_training(self):
        self.W_s = []
        self.H_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            W, H = self.sklearn_pretrain(i)
            self.W_s.append(W)
            self.H_s.append(H)


    def training(self):
        self.loss = []
        for iteration in tqdm(range(100), desc="Training pass: ", leave=True):
            for i in range(self.p):
                self.update_W(i)
                self.update_H(i)
            self.S = self.W_s[1].dot(self.H_s[1]).dot(self.H_s[0])

            if True:
                self.calculate_cost(iteration)

        return self.S


    def update_W(self, i):
        if i == 0:
            R1 = self.A.dot(self.H_s[0].T)
            R2 = self.W_s[0].dot(self.H_s[0]).dot(self.H_s[0].T)
            self.W_s[0] = (self.W_s[0]*R1)/np.maximum(R2, 10**-10)   
        elif i == 1:
            R1 = self.A.dot(self.H_s[0].T).dot(self.H_s[1].T)
            R2 = self.W_s[1].dot(self.H_s[1]).dot(self.H_s[0]).dot(self.H_s[0].T).dot(self.H_s[1].T)
            self.W_s[1] = (self.W_s[1]*R1)/np.maximum(R2, 10**-10)
        else:
            R1 = self.A.dot(self.H_s[0].T).dot(self.H_s[1].T).dot(self.H_s[2].T)
            R2 = self.W_s[2].dot(self.H_s[2]).dot(self.H_s[1]).dot(self.H_s[0]).dot(self.H_s[0].T).dot(self.H_s[1].T).dot(self.H_s[2].T)
            self.W_s[2] = (self.W_s[2]*R1)/np.maximum(R2, 10**-10)
            

    def update_H(self, i):
        if i == 0:
            R1 = self.W_s[0].T.dot(self.A)
            R2 = self.W_s[0].T.dot(self.W_s[0]).dot(self.H_s[0])
            self.H_s[0] = (self.H_s[0]*R1)/np.maximum(R2, 10**-10)
        elif i == 1:
            R1 = self.W_s[1].T.dot(self.A).dot(self.H_s[0].T)
            R2 = self.W_s[1].T.dot(self.W_s[1]).dot(self.H_s[1]).dot(self.H_s[0]).dot(self.H_s[0].T)
            self.H_s[1] = (self.H_s[1]*R1)/np.maximum(R2, 10**-10)
        else:
            R1 = self.W_s[2].T.dot(self.A).dot(self.H_s[0].T).dot(self.H_s[1].T)
            R2 = self.W_s[2].T.dot(self.W_s[2]).dot(self.H_s[2]).dot(self.H_s[1]).dot(self.H_s[0]).dot(self.H_s[0].T).dot(self.H_s[1].T)
            self.H_s[2] = (self.H_s[2]*R1)/np.maximum(R2, 10**-10)


    def calculate_cost(self, i):
        reconstruction_loss = 1/2 * np.linalg.norm(self.A-self.S, ord="fro")**2
        self.loss.append([i+1,reconstruction_loss])
