import numpy as np
import networkx as nx
import copy
import scipy
import networkx as nx
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from node2vec import Node2Vec
from DNMFA import dnmfa
from DNMFA_F import dnmfa_f
from DNMFA_R import dnmfa_r
from EDNMFA import ednmfa
from DNMF import dnmf
from DANMF import danmf
from FSSDNMF import fssdnmf
from FSSDNMF_2layer import fssdnmf_2layer
from NMF_A1 import nmfa1
from NMF_D1 import nmfd1
from DGLP import dglp
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from sklearn.metrics import accuracy_score

class predict:
    def __init__(self, train_G, positive_datas, negative_datas):
        self.train_G = train_G
        self.positive_datas = positive_datas
        self.negative_datas = negative_datas
        self.L = len(self.positive_datas)

        self.train_nodes = list(self.train_G.nodes())
        self.A = nx.adjacency_matrix(self.train_G).toarray()


    def Precision_score(self, S):
        edges_scores = {}

        for i in self.positive_datas:
            edges_scores[i] = S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]]

        for j in self.negative_datas:
            edges_scores[j] = S[[self.train_nodes.index(j[0])],[self.train_nodes.index(j[1])]]

        sorted_node_pairs = [key for key, value in sorted(edges_scores.items(), key=lambda x: x[1], reverse=True)]

        predict_pair = sorted_node_pairs[:self.L]
        Lr = 0

        for pair in predict_pair:
            if pair in self.positive_datas:
                Lr += 1
        score = Lr/self.L
        
        return score


    def predict_based_DNMFA(self):
        start_time = time.time()
        DNMFA_model = dnmfa(self.train_G)
        S = DNMFA_model.training()
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        DNMFA_AUC_score = roc_auc_score(y_true, y_score)
        DNMFA_PR_score = self.Precision_score(S)

        return DNMFA_AUC_score, DNMFA_PR_score


    def predict_based_DNMFAF(self):
        start_time = time.time()
        DNMFAF_model = dnmfa_f(self.train_G)
        S = DNMFAF_model.training()
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        DNMFAF_AUC_score = roc_auc_score(y_true, y_score)
        DNMFAF_PR_score = self.Precision_score(S)

        return DNMFAF_AUC_score, DNMFAF_PR_score


    def predict_based_DNMFAR(self):
        DNMFAR_model = dnmfa_r(self.train_G)
        S = DNMFAR_model.training()

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        DNMFAR_AUC_score = roc_auc_score(y_true, y_score)
        DNMFAR_PR_score = self.Precision_score(S)


        return DNMFAR_AUC_score, DNMFAR_PR_score


    def predict_based_EDNMFA(self):
        EDNMFA_model = ednmfa(self.train_G)
        S = EDNMFA_model.training()

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        EDNMFA_AUC_score = roc_auc_score(y_true, y_score)
        EDNMFA_PR_score = self.Precision_score(S)

        return EDNMFA_AUC_score, EDNMFA_PR_score


    def predict_based_DANMF(self):
        start_time = time.time()
        DANMF_model = danmf(self.train_G)
        DANMF_model.pre_training()
        S = DANMF_model.training()
        end_time = time.time()
        print(end_time - start_time)


        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        DANMF_AUC_score = roc_auc_score(y_true, y_score)
        DANMF_PR_score = self.Precision_score(S)

        return DANMF_AUC_score, DANMF_PR_score


    def predict_based_FSSDNMF(self):
        start_time = time.time()
        X = nx.adjacency_matrix(self.train_G)
        FSSDNMF_model = fssdnmf(X)
        FSSDNMF_model.pre_training()
        S = FSSDNMF_model.training()
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        FSSDNMF_AUC_score = roc_auc_score(y_true, y_score)
        FSSDNMF_PR_score = self.Precision_score(S)

        return FSSDNMF_AUC_score, FSSDNMF_PR_score


    def predict_based_DNMF(self):
        start_time = time.time()
        A = nx.adjacency_matrix(self.train_G)
        DNMF_model = dnmf(A)
        DNMF_model.pre_training()
        S = DNMF_model.training()
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        DNMF_AUC_score = roc_auc_score(y_true, y_score)
        DNMF_PR_score = self.Precision_score(S)

        return DNMF_AUC_score, DNMF_PR_score


    def predict_based_NMF(self):
        start_time = time.time()
        A = nx.adjacency_matrix(self.train_G)

        nmf_model = NMF(n_components=25,
                   init="random",
                   random_state=42,
                   max_iter=300)
        W = nmf_model.fit_transform(csr_matrix(A))
        H = nmf_model.components_
        S = np.dot(W, H)
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        NMF_AUC_score = roc_auc_score(y_true, y_score)
        NMF_PR_score = self.Precision_score(S)

        return NMF_AUC_score, NMF_PR_score


    def predict_based_NMFA1(self):
        start_time = time.time()
        NMFA1_model = nmfa1(self.train_G)
        S = NMFA1_model.training()
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        NMFA1_AUC_score = roc_auc_score(y_true, y_score)
        NMFA1_PR_score = self.Precision_score(S)

        return NMFA1_AUC_score, NMFA1_PR_score


    def predict_based_NMFD1(self):
        start_time = time.time()
        NMFD1_model = nmfd1(self.train_G)
        S = NMFD1_model.training()
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,S[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        NMFD1_AUC_score = roc_auc_score(y_true, y_score)
        NMFD1_PR_score = self.Precision_score(S)

        return NMFD1_AUC_score, NMFD1_PR_score


    def predict_based_CN(self):
        start_time = time.time()
        A = self.A.copy()
        Matrix_similarity = np.dot(A,A)
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        print(y_true)
        print(y_score)

        CN_AUC_score = roc_auc_score(y_true, y_score)
        CN_PR_score = self.Precision_score(Matrix_similarity)

        return CN_AUC_score, CN_PR_score


    def predict_based_DGLP(self):
        start_time = time.time()
        dglp_model = dglp(self.train_G)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,dglp_model.training(i[0], i[1]))
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,dglp_model.training(i[0], i[1]))
            y_true = np.append(y_true,0)

        end_time = time.time()
        print(end_time - start_time)

        DGLP_AUC_score = roc_auc_score(y_true, y_score)
        DGLP_PR_score = average_precision_score(y_true, y_score)

        return DGLP_AUC_score, DGLP_PR_score


    def predict_based_LP(self):
        start_time = time.time()
        A = self.A.copy()
        Matrix_similarity = np.dot(A,A)
        Matrix_LP = np.dot(np.dot(A,A),A) 
        Matrix_similarity = np.dot(Matrix_similarity,Matrix_LP)
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        LP_AUC_score = roc_auc_score(y_true, y_score)
        LP_PR_score = self.Precision_score(Matrix_similarity)

        return LP_AUC_score, LP_PR_score


    def predict_based_Katz(self):
        start_time = time.time()
        A = self.A.copy()
        Matrix_EYE = np.eye(A.shape[0])
        Temp = Matrix_EYE - A * 0.01
        Matrix_similarity = np.linalg.inv(Temp)
        Matrix_similarity = Matrix_similarity - Matrix_EYE
        end_time = time.time()
        print(end_time - start_time)


        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        Katz_AUC_score = roc_auc_score(y_true, y_score)
        Katz_PR_score = self.Precision_score(Matrix_similarity)

        return Katz_AUC_score, Katz_PR_score


    def predict_based_Cos(self):
        start_time = time.time()
        A = self.A.copy()
        Matrix_D = np.diag(sum(A))
        Matrix_Laplacian = Matrix_D - A
        INV_Matrix_Laplacian  = scipy.linalg.pinv(Matrix_Laplacian)
        end_time = time.time()
        print(end_time - start_time)

        Array_Diag = np.diag(INV_Matrix_Laplacian)
        Matrix_ONE = np.ones([A.shape[0],A.shape[0]])
        Matrix_Diag = Array_Diag * Matrix_ONE

        Matrix_similarity = INV_Matrix_Laplacian/((Matrix_Diag * Matrix_Diag.T) ** 0.5)
        Matrix_similarity = np.nan_to_num(Matrix_similarity)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        Katz_AUC_score = roc_auc_score(y_true, y_score)
        Katz_PR_score = self.Precision_score(Matrix_similarity)

        return Katz_AUC_score, Katz_PR_score


    def predict_based_HDI(self):
        A = self.A.copy()
        Matrix_similarity = np.dot(A,A)
        deg_row = sum(A)
        deg_row.shape = (deg_row.shape[0],1)
        deg_row_T = deg_row.T
        tempdeg = np.maximum(deg_row,deg_row_T)
        Matrix_similarity = Matrix_similarity / tempdeg

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        HDI_AUC_score = roc_auc_score(y_true, y_score)
        HDI_PR_score = average_precision_score(y_true, y_score)

        return HDI_AUC_score, HDI_PR_score


    def predict_based_HPI(self):
        A = self.A.copy()
        Matrix_similarity = np.dot(A,A)
        deg_row = sum(A)
        deg_row.shape = (deg_row.shape[0],1)
        deg_row_T = deg_row.T
        tempdeg = np.minimum(deg_row,deg_row_T)
        Matrix_similarity = Matrix_similarity / tempdeg

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        HPI_AUC_score = roc_auc_score(y_true, y_score)
        HPI_PR_score = average_precision_score(y_true, y_score)

        return HPI_AUC_score, HPI_PR_score


    def predict_based_PA(self):
        A = self.A.copy()
        deg_row = sum(A)
        deg_row.shape = (deg_row.shape[0],1)
        deg_row_T = deg_row.T
        Matrix_similarity = np.dot(deg_row,deg_row_T)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        PA_AUC_score = roc_auc_score(y_true, y_score)
        PA_PR_score = average_precision_score(y_true, y_score)

        return PA_AUC_score, PA_PR_score


    def predict_based_AA(self):
        start_time = time.time()
        A = self.A.copy()
        logTrain = np.log(sum(A))
        logTrain = np.nan_to_num(logTrain)
        logTrain.shape = (logTrain.shape[0],1)
        MatrixAdjacency_Train_Log = A / logTrain
        MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)
        Matrix_similarity = np.dot(A,MatrixAdjacency_Train_Log)
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        AA_AUC_score = roc_auc_score(y_true, y_score)
        AA_PR_score = self.Precision_score(Matrix_similarity)

        return AA_AUC_score, AA_PR_score


    def predict_based_RA(self):
        start_time = time.time()
        A = self.A.copy()
        RA_Train = sum(A)
        RA_Train.shape = (RA_Train.shape[0],1)
        MatrixAdjacency_Train_Log = A / RA_Train
        MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)
        Matrix_similarity = np.dot(A,MatrixAdjacency_Train_Log)
        end_time = time.time()
        print(end_time - start_time)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        RA_AUC_score = roc_auc_score(y_true, y_score)
        RA_PR_score = self.Precision_score(Matrix_similarity)

        return RA_AUC_score, RA_PR_score


    def predict_based_Jaccavrd(self):
        A = self.A.copy()
        Matrix_similarity = np.dot(A,A)

        deg_row = sum(A)
        deg_row.shape = (deg_row.shape[0],1)
        deg_row_T = deg_row.T
        tempdeg = deg_row + deg_row_T
        temp = tempdeg - Matrix_similarity

        Matrix_similarity = Matrix_similarity / temp

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            y_score = np.append(y_score,Matrix_similarity[[self.train_nodes.index(i[0])],[self.train_nodes.index(i[1])]])
            y_true = np.append(y_true,0)

        Jaccavrd_AUC_score = roc_auc_score(y_true, y_score)
        Jaccavrd_PR_score = average_precision_score(y_true, y_score)

        return Jaccavrd_AUC_score, Jaccavrd_PR_score


    def predict_based_DeepWalk(self):
        node2vec = Node2Vec(self.train_G, dimensions=128, walk_length=40, num_walks=5, workers=2, p=1, q=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=32)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            node_positive_vector1 = model.wv[i[0]]
            node_positive_vector2 = model.wv[i[1]]
            normalization_positive_vec1 = node_positive_vector1 / (np.sqrt(np.sum(node_positive_vector1 ** 2)))
            normalization_positive_vec2 = node_positive_vector2 / (np.sqrt(np.sum(node_positive_vector2 ** 2)))
            y_score = np.append(y_score,np.dot(normalization_positive_vec1, normalization_positive_vec2))
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            node_negative_vector1 = model.wv[i[0]]
            node_negative_vector2 = model.wv[i[1]]
            normalization_negative_vec1 = node_negative_vector1 / (np.sqrt(np.sum(node_negative_vector1 ** 2)))
            normalization_negative_vec2 = node_negative_vector2 / (np.sqrt(np.sum(node_negative_vector2 ** 2)))
            y_score = np.append(y_score,np.dot(normalization_negative_vec1, normalization_negative_vec2))
            y_true = np.append(y_true,0)

        DeepWalk_AUC_score = roc_auc_score(y_true, y_score)
        DeepWalk_PR_score = average_precision_score(y_true, y_score)

        return DeepWalk_AUC_score, DeepWalk_PR_score
    

    def predict_based_Node2Vec(self):
        node2vec = Node2Vec(self.train_G, dimensions=128, walk_length=40, num_walks=5, workers=2, p=0.25, q=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=32)

        y_true = np.array([], dtype=int)
        y_score = np.array([], dtype=int)

        for i in self.positive_datas:
            node_positive_vector1 = model.wv[i[0]]
            node_positive_vector2 = model.wv[i[1]]
            normalization_positive_vec1 = node_positive_vector1 / (np.sqrt(np.sum(node_positive_vector1 ** 2)))
            normalization_positive_vec2 = node_positive_vector2 / (np.sqrt(np.sum(node_positive_vector2 ** 2)))
            y_score = np.append(y_score,np.dot(normalization_positive_vec1, normalization_positive_vec2))
            y_true = np.append(y_true,1)

        for i in self.negative_datas:
            node_negative_vector1 = model.wv[i[0]]
            node_negative_vector2 = model.wv[i[1]]
            normalization_negative_vec1 = node_negative_vector1 / (np.sqrt(np.sum(node_negative_vector1 ** 2)))
            normalization_negative_vec2 = node_negative_vector2 / (np.sqrt(np.sum(node_negative_vector2 ** 2)))
            y_score = np.append(y_score,np.dot(normalization_negative_vec1, normalization_negative_vec2))
            y_true = np.append(y_true,0)

        Node2Vec_AUC_score = roc_auc_score(y_true, y_score)
        Node2Vec_PR_score = average_precision_score(y_true, y_score)

        return Node2Vec_AUC_score, Node2Vec_PR_score
        

