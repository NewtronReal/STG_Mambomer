import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Callable
from Transformer.modules.GraphFeatures import GraphFeatures
from Transformer.modules.CentralityEncoder import CentralityEncoder

## Conv2D module as written in T-Graphormer
"""
The Conv2D class can be though as a fully connected linear layer for 
embedding the C dimensional features into d dimensions a kernel of size
[1,1](one pixel at a time) moves over TxN matrix with C channels, and the
Kernel has d dimensions which slides over TxN matrix converting the channels
to d. This process is repeated 2 times first with gelu activation function and
second without gelu activation. In short the FC is a replacement for 2 fully 
connected layers which transforms B,T,N,C->Linear->gelu(x)->Linear->output
"""
class Conv2D(nn.Module):
    def __init__(
            self,
            input_dims: int,#initial number of features,the D in datashape 
            output_dims: int,
            kernel_size: Union[tuple, list],
            stride: Union[tuple, list] = (1, 1),
            use_bias: bool = False,
            activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = F.gelu,
    ):
        super(Conv2D, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm2d(output_dims)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, use_bias):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList(
            [
                Conv2D(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    use_bias=use_bias,
                    activation=activation,
                )
                for input_dim, num_unit, activation in zip(
                input_dims, units, activations
            )
            ]
        )

    def forward(self, x):
        x = x.contiguous().permute(0, 3, 2, 1)
        for conv in self.convs:
            x = conv(x)
        x = x.contiguous().permute(0, 3, 2, 1)
        return x
    
class GraphNodeFeature(nn.Module):
    def __init__(self,
                 N,
                 C,
                 d,
                 graph:GraphFeatures,
                 ):
        super().__init__()
        self.N = N
        self.C = C
        self.d = d
        self.graph = graph
        self.c_enc = CentralityEncoder(graph.max_in_degree,graph.max_out_degree,d)
        self.conv = FC(
            input_dims=[C,d],
            units=[d,d],
            activations=[nn.GELU(),None],
            use_bias = False
        )
    def forward(self,x):
        x =self.conv(x)
        x = self.c_enc(x,self.graph.in_degree,self.graph.out_degree)
        return x
