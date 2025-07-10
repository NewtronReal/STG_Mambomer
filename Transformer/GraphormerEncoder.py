import torch
import torch.nn as nn
from modules.CentralityEncoder import CentralityEncoder
from modules.GraphFeatures import GraphFeatures
from modules.MultiLayerConvolution import FC

"""
Basic layer combining Transformer, Positional Encoding(Token Positions), Centrality Encoding(Encoding Degrees)
MultiLayer Convolution and Multihead attension(without Q,K,V output)
"""

class GraphormerEncoder(nn.Module):
    def __init__(self,
                 N:int=307,
                 C:int=1,
                 T:int=12,
                 d:int=6,
                 hno:int=2,
                 layer_no:int=4,
                 graph:GraphFeatures=GraphFeatures()
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.C = C
        self.T = T
        self.d = d
        self.graph =graph
        assert d%hno == 0, "hno should be a perfect divisor of d"
        self.hno = hno
        self.conv = FC(
            input_dims=[C,d],
            units=[d,d],
            activations=[nn.GELU(),None],
            use_bias = False
        )
        self.c_encoder = CentralityEncoder(graph.max_in_degree,graph.max_out_degree,d)
        self.time_embeddings = nn.Embedding(T,d).to(self.device)
        self.node_embeddings = nn.Embedding(N,d).to(self.device)
        
        self.input_projection = nn.Linear(d,d).to(self.device)
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d,nhead=hno,batch_first=True).to(self.device)
        self.encoder = nn.TransformerEncoder(self.transformer_layer,num_layers = layer_no).to(self.device)
        
        self.out_projection = nn.Linear(d,C).to(self.device)
        
    def _generate_casual_mask(self):#prevents older timestamps attending to newer timestamps but graph nodes at a particular time stamp can attend each other
        temp_ids = torch.arange(self.T).repeat_interleave(self.N)
        mask = temp_ids.unsqueeze(0)<temp_ids.unsqueeze(1)
        return mask.float().masked_fill(mask,float('-inf')).to(self.device)
    
    def forward(self,x):
        B,T,N,C = x.shape
        assert (T,N,C) == (self.T,self.N,self.C),"Dimensions doesn't match"
        x=self.conv(x)#x->Linear(C,d)->gelu->Linear(d,d)
        x=self.c_encoder(x,self.graph.in_degree,self.graph.out_degree)#Centrality encoding each node gets importance with respect to their in_degree/out_degree
        x=x.contiguous().view(B,T*N,self.d)#temporal flattening
        temp_ids = torch.arange(T).unsqueeze(0).expand(N,T).reshape(-1).to(self.device)#token generation for positional encoding
        spatial_ids = torch.arange(N).unsqueeze(1).expand(N,T).reshape(-1).to(self.device)
        
        temp_emb = self.time_embeddings(temp_ids)#generate embeddings for different time step and different node
        spatial_emb = self.node_embeddings(spatial_ids)
        
        pos_enc = temp_emb+spatial_emb
        
        pos_enc = pos_enc.unsqueeze(0).expand(B,-1,-1)
        
        x=self.input_projection(x)+pos_enc#x=x->Linear(d,d)+posenc
        
        mask = self._generate_casual_mask()
        encodings = self.encoder(x,mask=mask)#x_masked
        
        out = self.out_projection(encodings)
        out = out.view(B,T,N,C)
        
        return out