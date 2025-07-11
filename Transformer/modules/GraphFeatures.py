import torch
import torch.nn
from Transformer.data.data import get_graph_info
from Transformer.algo.graph_algo import floyd_warshall
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import dense_to_sparse



class GraphFeatures():
    def __init__(self,adj_path="adj.npz",device='cpu',clip=50):
        self.device = device
        a = self.get_graph_info(adj_path)
        self.adj = a[0]
        self.in_degree = a[1]
        self.out_degree = a[2]
        self.max_in_degree = a[3]
        self.max_out_degree = a[4]
        self.spd,self.path = floyd_warshall(self.adj,clip)
    def get_graph_info(self,adj_path:str = "adj.npz",normalize:bool=True,scale=10):
        adj = torch.tensor(sp.load_npz(adj_path).toarray(),dtype=torch.float32).to(self.device)
        adj[adj!=0] = 1/adj[adj!=0]
        if normalize:
            maxval = torch.max(torch.max(adj))
            adj=((adj)/maxval)*scale
        in_degree = (adj!=0).sum(dim=0)
        out_degree = (adj!=0).sum(dim=1)
        max_in_degree = max(in_degree)
        max_out_degree = max(out_degree)
        return adj,in_degree,out_degree,max_in_degree,max_out_degree



