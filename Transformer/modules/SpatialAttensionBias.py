import torch
import torch.nn as nn
from Transformer.modules.GraphFeatures import *

#For now we are not considering cls token
class SpatialAttensionBias(nn.Module):
    def __init__(self,graph:GraphFeatures,d,S,hno,max_dist=50):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hno = hno
        self.graph = graph
        self.d = d
        self.sp_enc = nn.Embedding(S+1,hno,padding_idx=0).to(self.device)
    def forward(self,x):
        B,T,N,D = x.shape
        attn_bias = torch.zeros([B,N+1,N+1],dtype = torch.float).to(self.device)
        graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1,self.hno,1,1)
        sp_bias = self.graph.spd.long().unsqueeze(0).repeat(B,1,1)
        sp_bias = self.sp_enc(sp_bias).contiguous().permute(0,3,1,2)
        graph_attn_bias[:,:,1:,1:] = graph_attn_bias[:,:,1:,1:]+sp_bias
        graph_attn_bias = graph_attn_bias +attn_bias.unsqueeze(1)
        return graph_attn_bias
        