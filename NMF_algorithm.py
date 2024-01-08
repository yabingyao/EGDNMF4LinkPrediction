import numpy as np
import networkx as nx
import random


class nmf:
    def __init__(self, A):
        self.A = A.toarray()
        self.k = 25
        self.m, self.n = A.shape


    def training(self):
        self.W = np.random.random((self.m, self.k))
        self.H = np.random.random((self.k, self.n))

        for iteration in range(100):
            self.update_W()
            self.update_H()

        self.S = self.W.dot(self.H)
        return self.S


    def update_W(self):
        R1 = self.A.dot(self.H.T)
        R2 = self.W.dot(self.H).dot(self.H.T)
        self.W = (self.W*R1)/np.maximum(R2, 10**-10)   

    def update_H(self):
        R1 = self.W.T.dot(self.A)
        R2 = self.W.T.dot(self.W).dot(self.H)
        self.H = (self.H*R1)/np.maximum(R2, 10**-10)