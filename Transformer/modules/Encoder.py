from typing import Optional,Union
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from Transformer.modules.MultiLayerConvolution import GraphNodeFeature
from Transformer.modules.SpatialAttensionBias import SpatialAttensionBias
from Transformer.modules.EncodingLayer import EncodingLayer
from Transformer.modules.GraphFeatures import GraphFeatures

class Encoder(nn.Module):
    def __init__(self,
                 C,
                 N,
                 graph:GraphFeatures,
                 enclayers=12,
                 d=768,
                 ffno=768,
                 hno=32,
                 dropout=0.1,
                 S=50
                 ):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.hno = hno
        self.gnf = GraphNodeFeature(N,C,d,graph)
        self.sp_bias = SpatialAttensionBias(graph,d,S,hno)
        self.emb_layer_norm = LayerNorm(d,eps=1e-8)
        self.layers = nn.ModuleList([])
        self.layers.extend([EncodingLayer(d,ffno,hno,dropout) for _ in range(enclayers)])
        
    def compute_attn_bias(self,x):
        attn_bias = self.sp_bias(x)
        return attn_bias
    def compute_mods(self,x):
        x=self.gnf(x)
        attn_bias = self.sp_bias(x)
        return x,attn_bias
    def forward_layers(self,x,attn_bias):
        x = x.contiguous().transpose(0,1)
        for layer in self.layers:
            x,attn = layer(
                x,attn_bias
            )
        return x,attn
    def forward(self,x,attn_bias):
        B,L,d = x.shape
        x = self.emb_layer_norm(x)
        x,attn = self.forward_layers(x,attn_bias)
        return x,attn