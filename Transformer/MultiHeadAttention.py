import torch
import torch.nn as nn
from torch.nn.functional import softmax

"""
Custom MHA module inspired from T-Graphormer will be replacing GraphormerEncoder
"""
class MHA(nn.Module):
    def __init__(self,input:int,embedd:int,hno:int,output:int):
        super().__init__()
        self.hno = hno
        self.hsz = embedd // hno
        assert self.hno*self.hsz == embedd, "Wrong hno, hsz is not an integer"
        self.W_Q = nn.Linear(input,embedd)
        self.W_K = nn.Linear(input,embedd)
        self.W_V = nn.Linear(input,embedd)
        self.output_projection = nn.Linear(embedd,output)
    def forward(self,x):
        bsz,slen,d=x.size()
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        
        q = q.view(bsz,slen,self.hno,self.hsz).transpose(1,2)
        k = k.view(bsz,slen,self.hno,self.hsz).transpose(1,2)
        v = v.view(bsz,slen,self.hno,self.hsz).transpose(1,2)
        mask = torch.tril(torch.ones(slen, slen,dtype=bool)).unsqueeze(0).unsqueeze(0)
        
        attension = torch.matmul(q, k.transpose(-2, -1))
        attension = attension/(q.shape[-1]**.5)
        attension = attension.masked_fill(~mask, float('-inf'))
        attension = softmax(attension,dim=-1)
        
        value = torch.matmul(attension,v)
        value = value.transpose(-2, -1).reshape(bsz, slen, -1)
        out = self.output_projection(value)
        return value