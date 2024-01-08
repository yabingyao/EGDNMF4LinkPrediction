# EGDNMF
Yao Y, He Y, Huang Z, et al. [Deep non-negative matrix factorization with edge generator for link prediction in complex networks](https://link.springer.com/article/10.1007/s10489-023-05211-1#citeas) [J]. Applied Intelligence, 2023: 1-22.
# Abstract
Link prediction aims to infer missing links or predict future links based on observed topology or attribute information in the network. Many link prediction methods based on non-negative matrix factorization (NMF) have been proposed to solve prediction problem. However, due to the sparsity of real networks, the observed topology information is probably very limited, which affects the performance of existing link prediction methods. In this paper, we utilize Deep Non-negative Matrix Factorization (DNMF) models with Edge Generator to address the network sparsity problem and propose link prediction methods EG-DNMF and EG-FDNMF. Under the framework of DNMF, several representative potential edges are incorporated so as to reconstruct the original network for link prediction. Specifically, in order to explore the potential structural features of the network in a more fine-grained manner, we first divide the original network into three sub-networks. Then, the DNMF models are employed to mine complex and nonlinear interaction relationships in sub-networks, thereby guiding the network reconstruction process. Finally, the NMF algorithm is applied on the reconstructed original network for link prediction. Experiment results on 12 different networks show that our methods have comparable performance with respect to 13 representative link prediction methods which include 6 NMF/DNMF-based approaches and 7 heuristic-based approaches. In addition, experiments also show that the sub-networks after partitioning are beneficial for capturing the underlying features of the network.  
![EGDNMF](https://github.com/hyy177/EGDNMF/blob/main/EGDNMF.jpg)
# Citing
If you find _EGDNMF_ useful for your research, please consider citing the following paper:  
  
@article{yao2023deep,  
  title={Deep non-negative matrix factorization with edge generator for link prediction in complex networks},  
  author={Yao, Yabing and He, Yangyang and Huang, Zhentian and Xu, Zhipeng and Yang, Fan and Tang, Jianxin and Gao, Kai},  
  journal={Applied Intelligence},  
  pages={1--22},  
  year={2023},  
  publisher={Springer},  
  doi={[https://doi.org/10.1007/s10489-023-05211-1](https://doi.org/10.1007/s10489-023-05211-1)},  
  url={[https://link.springer.com/article/10.1007/s10489-023-05211-1#citeas](https://link.springer.com/article/10.1007/s10489-023-05211-1#citeas)}  
}
