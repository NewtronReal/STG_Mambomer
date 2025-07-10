from modules.MAE_Mamba import MAE_Mamba
from modules.GraphFeatures import GraphFeatures
from data.data import PrepareDataset
import torch
import pandas as pd


data = pd.read_csv("pems04_flow.csv")
data = PrepareDataset(data)


# graph = GraphFeatures()
# # graph.adj = torch.ones(306,306).long()
# # graph.spd = torch.ones(306,306)
# # graph.in_degree = torch.ones(306).long()
# # graph.out_degree = torch.ones(306).long()
# g = MAE_Mamba(graph=graph,N=307,T=3,hno=8)
# a = g(torch.rand(1,3,307,1),.5)
# print(a[2])