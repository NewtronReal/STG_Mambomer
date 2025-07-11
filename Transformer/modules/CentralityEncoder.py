import torch
import torch.nn as nn

class CentralityEncoder(nn.Module):
    def __init__(self,max_in_degree,max_out_degree,embedding):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding = embedding
        self.z_in = nn.Embedding(max_in_degree+1,embedding).to(self.device)
        self.z_out = nn.Embedding(max_out_degree+1,embedding).to(self.device)
    def forward(self,x,in_degree,out_degree):
        _,_,N,d = x.shape
        assert in_degree.shape[0] == N and out_degree.shape[0] == N,"Node dimensions does not match"
        assert d==self.embedding, "Feature embedd dimensions does not match"
        centrality = self.z_in(in_degree)+self.z_out(out_degree)
        x+=centrality
        return x