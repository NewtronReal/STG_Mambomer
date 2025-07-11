from Transformer.modules.MAE_Mamba import MAE_Mamba
from Transformer.modules.GraphFeatures import GraphFeatures
from Transformer.data.data import PrepareDataset,get_device
import torch
import pandas as pd

data = pd.read_csv("Transformer/pems04_flow.csv")
data = PrepareDataset(data,BATCH_SIZE=3)
traind = data[0]

one_batch = next(iter(traind))[0]

B,W,m,N,D = one_batch.shape
compound = one_batch.view(B*W,m,N,D)

graph = GraphFeatures(adj_path = "Transformer/adj.npz",device=get_device())
# graph.adj = torch.ones(306,306).long()
# graph.spd = torch.ones(306,306)
# graph.in_degree = torch.ones(306).long()
# graph.out_degree = torch.ones(306).long()
g = MAE_Mamba(graph=graph,N=307,T=3,hno=8).to(get_device())
a = g(compound,.5)
print(a[2])