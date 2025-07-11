import torch
import torch.nn as nn
from Transformer.modules.MHA import MHA

class EncodingLayer(nn.Module):
    def __init__(self,
                 d=32,
                 ffno=64,
                 hno=8,
                 dropout = 0.1,
                 pre_layernorm=True
                 ):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.d = d
        self.hno = hno
        self.dropout = dropout
        self.pre_layernorm = pre_layernorm
        
        self.dropout_module = nn.Dropout(dropout).to(self.device)
        self.activation_dropout = nn.Dropout(dropout).to(self.device)
        self.activation = nn.GELU().to(self.device)
        self.self_attn = MHA(d,d,hno,d).to(self.device)
        self.self_attn_layer_norm = nn.LayerNorm(d,eps=1e-8).to(self.device)
        self.fc1 = nn.Linear(d,ffno).to(self.device)
        self.fc2 = nn.Linear(ffno,d).to(self.device)
        self.final_layer_norm = nn.LayerNorm(d).to(self.device)
    def forward(self,
                x:torch.Tensor,
                attn_bias=None,
                mask=None):
        residual = x
        x=self.self_attn_layer_norm(x)
        x,attn = self.self_attn(x,attn_bias)
        x=self.dropout_module(x)
        
        x = residual+x
        
        residual = x
        x=self.final_layer_norm(x)
        x=self.activation(self.fc1(x))
        x= self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual+x
        return x,attn

