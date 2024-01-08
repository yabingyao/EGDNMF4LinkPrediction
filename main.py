import numpy as np
import pandas as pd
import networkx as nx
from train_test_data import Divide_data
from link_prediction import predict

#path = "E:\\datasets\\all\\utm_300_2191.txt"
path = "./datasets/utm_300_2191.txt"
divide_rate = 0.9
number = 20

#用于存放每一轮的分数(AUC)
DNMFA_AUCscores = []
DNMFAF_AUCscores = []
DNMFAR_AUCscores = []
EDNMFA_AUCscores = []

DANMF_AUCscores = []
FSSDNMF_AUCscores = []
DNMF_AUCscores = []
NMF_AUCscores = []
NMFA1_AUCscores = []
NMFD1_AUCscores = []

CN_AUCscores = []
AA_AUCscores = []
RA_AUCscores = []
LP_AUCscores = []
Katz_AUCscores = []
Cos_AUCscores = []
DGLP_AUCscores = []

#用于存放每一轮的分数(PR)
DNMFA_PRscores = []
DNMFAF_PRscores = []
DNMFAR_PRscores = []
EDNMFA_PRscores = []

DANMF_PRscores = []
FSSDNMF_PRscores = []
DNMF_PRscores = []
NMF_PRscores = []
NMFA1_PRscores = []
NMFD1_PRscores = []

CN_PRscores = []
AA_PRscores = []
RA_PRscores = []
LP_PRscores = []
Katz_PRscores = []
Cos_PRscores = []
DGLP_PRscores = []

for i in range(number):
    train_test_model = Divide_data(path, divide_rate)  
    train_G, positive_datas, negative_datas = train_test_model.divide_train_test()

    predict_model = predict(train_G, positive_datas, negative_datas)

    # 自己的方法(4)
    DNMFA_AUC_score, DNMFA_PR_score = predict_model.predict_based_DNMFA()
    DNMFA_AUCscores.append(DNMFA_AUC_score)
    DNMFA_PRscores.append(DNMFA_PR_score)

    DNMFAF_AUC_score, DNMFAF_PR_score = predict_model.predict_based_DNMFAF()
    DNMFAF_AUCscores.append(DNMFAF_AUC_score)
    DNMFAF_PRscores.append(DNMFAF_PR_score)

    DNMFAR_AUC_score, DNMFAR_PR_score = predict_model.predict_based_DNMFAR()
    DNMFAR_AUCscores.append(DNMFAR_AUC_score)
    DNMFAR_PRscores.append(DNMFAR_PR_score)

    EDNMFA_AUC_score, EDNMFA_PR_score = predict_model.predict_based_EDNMFA()
    EDNMFA_AUCscores.append(EDNMFA_AUC_score)
    EDNMFA_PRscores.append(EDNMFA_PR_score)


    # 对比的DNMF/NMF方法(6)
    DANMF_AUC_score, DANMF_PR_score = predict_model.predict_based_DANMF()
    DANMF_AUCscores.append(DANMF_AUC_score)
    DANMF_PRscores.append(DANMF_PR_score)

    FSSDNMF_AUC_score, FSSDNMF_PR_score = predict_model.predict_based_FSSDNMF()
    FSSDNMF_AUCscores.append(FSSDNMF_AUC_score)
    FSSDNMF_PRscores.append(FSSDNMF_PR_score)

    DNMF_AUC_score, DNMF_PR_score = predict_model.predict_based_DNMF()
    DNMF_AUCscores.append(DNMF_AUC_score)
    DNMF_PRscores.append(DNMF_PR_score)


    NMF_AUC_score, NMF_PR_score = predict_model.predict_based_NMF()
    NMF_AUCscores.append(NMF_AUC_score)
    NMF_PRscores.append(NMF_PR_score)

    NMFA1_AUC_score, NMFA1_PR_score = predict_model.predict_based_NMFA1()
    NMFA1_AUCscores.append(NMFA1_AUC_score)
    NMFA1_PRscores.append(NMFA1_PR_score)

    NMFD1_AUC_score, NMFD1_PR_score = predict_model.predict_based_NMFD1()
    NMFD1_AUCscores.append(NMFD1_AUC_score)
    NMFD1_PRscores.append(NMFD1_PR_score)


    #对比的结构方法(11)
    CN_AUC_score, CN_PR_score = predict_model.predict_based_CN()
    CN_AUCscores.append(CN_AUC_score)
    CN_PRscores.append(CN_PR_score)

    AA_AUC_score, AA_PR_score = predict_model.predict_based_AA()
    AA_AUCscores.append(AA_AUC_score)
    AA_PRscores.append(AA_PR_score)

    RA_AUC_score, RA_PR_score = predict_model.predict_based_RA()
    RA_AUCscores.append(RA_AUC_score)
    RA_PRscores.append(RA_PR_score)

    LP_AUC_score, LP_PR_score = predict_model.predict_based_LP()
    LP_AUCscores.append(LP_AUC_score)
    LP_PRscores.append(LP_PR_score)

    Katz_AUC_score, Katz_PR_score = predict_model.predict_based_Katz()
    Katz_AUCscores.append(Katz_AUC_score)
    Katz_PRscores.append(Katz_PR_score)

    Cos_AUC_score, Cos_PR_score = predict_model.predict_based_Cos()
    Cos_AUCscores.append(Cos_AUC_score)
    Cos_PRscores.append(Cos_PR_score)

    DGLP_AUC_score, DGLP_PR_score = predict_model.predict_based_DGLP() 
    DGLP_AUCscores.append(DGLP_AUC_score)
    DGLP_PRscores.append(DGLP_PR_score)


AUC_scores = []
average_DNMFA_AUCscore = [sum(DNMFA_AUCscores) / number]
AUC_scores.append(average_DNMFA_AUCscore)

average_DNMFAF_AUCscore = [sum(DNMFAF_AUCscores) / number]
AUC_scores.append(average_DNMFAF_AUCscore)

average_DNMFAR_AUCscore = [sum(DNMFAR_AUCscores) / number]
AUC_scores.append(average_DNMFAR_AUCscore)

average_EDNMFA_AUCscore = [sum(EDNMFA_AUCscores) / number]
AUC_scores.append(average_EDNMFA_AUCscore)

average_DANMF_AUCscore = [sum(DANMF_AUCscores) / number]
AUC_scores.append(average_DANMF_AUCscore)

average_FSSDNMF_AUCscore = [sum(FSSDNMF_AUCscores) / number]
AUC_scores.append(average_FSSDNMF_AUCscore)

average_DNMF_AUCscore = [sum(DNMF_AUCscores) / number]
AUC_scores.append(average_DNMF_AUCscore)

average_NMF_AUCscore = [sum(NMF_AUCscores) / number]
AUC_scores.append(average_NMF_AUCscore)

average_NMFA1_AUCscore = [sum(NMFA1_AUCscores) / number]
AUC_scores.append(average_NMFA1_AUCscore)

average_NMFD1_AUCscore = [sum(NMFD1_AUCscores) / number]
AUC_scores.append(average_NMFD1_AUCscore)

average_CN_AUCscore = [sum(CN_AUCscores) / number]
AUC_scores.append(average_CN_AUCscore)

average_AA_AUCscore = [sum(AA_AUCscores) / number]
AUC_scores.append(average_AA_AUCscore)

average_RA_AUCscore = [sum(RA_AUCscores) / number]
AUC_scores.append(average_RA_AUCscore)

average_LP_AUCscore = [sum(LP_AUCscores) / number]
AUC_scores.append(average_LP_AUCscore)

average_Katz_AUCscore = [sum(Katz_AUCscores) / number]
AUC_scores.append(average_Katz_AUCscore)

average_Cos_AUCscore = [sum(Cos_AUCscores) / number]
AUC_scores.append(average_Cos_AUCscore)

average_DGLP_AUCscore = [sum(DGLP_AUCscores) / number]
AUC_scores.append(average_DGLP_AUCscore)

df_AUC = pd.DataFrame(AUC_scores)
df_AUC.to_excel('AUC_data.xlsx', index=True)


PR_scores = []
average_DNMFA_PRscore = [sum(DNMFA_PRscores) / number]
PR_scores.append(average_DNMFA_PRscore)

average_DNMFAF_PRscore = [sum(DNMFAF_PRscores) / number]
PR_scores.append(average_DNMFAF_PRscore)

average_DNMFAR_PRscore = [sum(DNMFAR_PRscores) / number]
PR_scores.append(average_DNMFAR_PRscore)

average_EDNMFA_PRscore = [sum(EDNMFA_PRscores) / number]
PR_scores.append(average_EDNMFA_PRscore)

average_DANMF_PRscore = [sum(DANMF_PRscores) / number]
PR_scores.append(average_DANMF_PRscore)

average_FSSDNMF_PRscore = [sum(FSSDNMF_PRscores) / number]
PR_scores.append(average_FSSDNMF_PRscore)

average_DNMF_PRscore = [sum(DNMF_PRscores) / number]
PR_scores.append(average_DNMF_PRscore)

average_NMF_PRscore = [sum(NMF_PRscores) / number]
PR_scores.append(average_NMF_PRscore)

average_NMFA1_PRscore = [sum(NMFA1_PRscores) / number]
PR_scores.append(average_NMFA1_PRscore)

average_NMFD1_PRscore = [sum(NMFD1_PRscores) / number]
PR_scores.append(average_NMFD1_PRscore)

average_CN_PRscore = [sum(CN_PRscores) / number]
PR_scores.append(average_CN_PRscore)

average_AA_PRscore = [sum(AA_PRscores) / number]
PR_scores.append(average_AA_PRscore)

average_RA_PRscore = [sum(RA_PRscores) / number]
PR_scores.append(average_RA_PRscore)

average_LP_PRscore = [sum(LP_PRscores) / number]
PR_scores.append(average_LP_PRscore)

average_Katz_PRscore = [sum(Katz_PRscores) / number]
PR_scores.append(average_Katz_PRscore)

average_Cos_PRscore = [sum(Cos_PRscores) / number]
PR_scores.append(average_Cos_PRscore)

average_DGLP_PRscore = [sum(DGLP_PRscores) / number]
PR_scores.append(average_DGLP_PRscore)

df_PR = pd.DataFrame(PR_scores)
df_PR.to_excel('PR_data.xlsx', index=True)