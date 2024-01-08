import random
import numpy as np
import networkx as nx
import copy

class Divide_data:
    def __init__(self, path, rate):
        self.rate = rate
        self.G = nx.read_edgelist(path)

    def divide_train_test(self):
        test_ratio = 1 - self.rate  # 计算选取多少比例的测试集

        all_edges = list(self.G.edges())
        negative_datas = list(nx.non_edges(self.G))
        test_length = int(round(test_ratio * len(all_edges)))  # 需要删除边的数量

        random.shuffle(all_edges) #将边列表打乱
        positive_datas = all_edges[:test_length] #测试集
        self.G.remove_edges_from(positive_datas) #训练集

        

        return self.G, positive_datas, negative_datas