import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math

"""
Custom MHA module inspired from T-Graphormer will be replacing GraphormerEncoder
"""
class MHA(nn.Module):
    def __init__(self,input:int,embedd:int,hno:int,output:int,dropout:int=0.1):
        super().__init__()
        self.hno = hno
        self.hsz = embedd // hno
        assert self.hno*self.hsz == embedd, "Wrong hno, hsz is not an integer"
        self.W_Q = nn.Linear(input,embedd,bias=True)
        self.W_K = nn.Linear(input,embedd,bias=True)
        self.W_V = nn.Linear(input,embedd,bias=True)
        self.output_projection = nn.Linear(embedd,output,bias=True)
        self.reset_parameters()
        self.dropout_module = nn.Dropout(dropout)
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_K.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_V.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)
    def forward(self,x,attn_bias,mask=None):
        L,B,d=x.size()
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        
        q = q.contiguous().view(-1,B*self.hno,self.hsz).transpose(0,1)
        k = k.contiguous().view(-1,B*self.hno,self.hsz).transpose(0,1)
        v = v.contiguous().view(-1,B*self.hno,self.hsz).transpose(0,1)
        #mask = torch.tril(torch.ones(L, L,dtype=bool)).unsqueeze(0).unsqueeze(0)
        
        attension = torch.bmm(q, k.contiguous().transpose(1, 2))
        attension = attension/(q.shape[-1]**.5)
        if attn_bias is not None:
            attension = attension+attn_bias.view(B*self.hno,L,L)
        if mask is not None:
            mask = mask.unsqueeze(0)
            attension +=mask
        # attension = attension.masked_fill(~mask, float('-inf'))
        attension_weights = softmax(attension,dim=-1).type_as(attension)
        attension = self.dropout_module(attension_weights)
        value = torch.bmm(attension,v)
        value = value.transpose(0, 1).contiguous().view(L, B, d)
        out = self.output_projection(value)
        return value,attension_weights