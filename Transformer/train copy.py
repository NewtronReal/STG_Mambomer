import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as utils
import torch.nn.functional as F
from typing import Union, Optional, Callable
import os
from data.data import PrepareDataset
from algo.graph_algo import floyd_warshall,find_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_path(path, i, j):
    if path[i, j] == 0:
        return [i, j]
    else:
        k = path[i, j]
        return find_path(path, i, k)[:-1] + find_path(path, k, j)

def floyd_warshall(adj):    
    if os.path.exists('floyd_warshall.npz'):
        mat = np.load("floyd_warshall.npz")
        dist,path=mat['a'],mat['b']
        return dist,path
    
    n = adj.shape[0]
    dist = adj.clone()
    path = torch.zeros_like(adj,dtype=torch.int64)
    dist[dist == 0] = float('inf')
    for i in range(n):
        dist[i][i] = 0
    count = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                count+=1
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        path[i,j] = k
                print(f"{count}/{n**3}")
    dist[dist>=510]=510
    np.savez('floyd_warshall',a=dist.numpy(),b=path.numpy())
    return dist,path

def PrepareDataset(speed_matrix, BATCH_SIZE=30, seq_len=12, pred_len=12, train_propotion=0.7, valid_propotion=0.1,limit=100000000):
    time_len = speed_matrix.shape[0]
    #max_speed = speed_matrix.max().max()
    #speed_matrix = speed_matrix / max_speed
    limit = min(time_len,limit)
    # MinMax Normalization Method.
    max_speed = speed_matrix.max().max()
    min_speed = speed_matrix.min().min()
    speed_matrix = (speed_matrix - min_speed)/(max_speed - min_speed)    

    speed_sequences, speed_labels = [], []
    for i in range(limit - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
    # Reshape labels to have the same second dimension as the sequences
    speed_labels = speed_labels.reshape(speed_labels.shape[0], seq_len, -1)

    # shuffle & split the dataset to training and testing sets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data).unsqueeze(-1).to(device), torch.Tensor(train_label).unsqueeze(-1).to(device)
    valid_data, valid_label = torch.Tensor(valid_data).unsqueeze(-1).to(device), torch.Tensor(valid_label).unsqueeze(-1).to(device)
    test_data, test_label = torch.Tensor(test_data).unsqueeze(-1).to(device), torch.Tensor(test_label).unsqueeze(-1).to(device)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print("Data Loaded")

    return train_dataloader, valid_dataloader, test_dataloader, max_speed


## Add adjacency matrix for encoding graph info

adj_path = "T-Graphormer/ST-Mambomer/Mamba/PEMS04/pems04_adj.npy"
adj = torch.tensor(np.load(adj_path))
in_degree = (adj!=0).sum(dim=0)
out_degree = (adj!=0).sum(dim=1)
max_in_degree = max(in_degree)
max_out_degree = max(out_degree)

class Conv2D(nn.Module):
    def __init__(
            self,
            input_dims: int,
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

class CentralityEncoder(nn.Module):
    def __init__(self,max_in_degree,max_out_degree,embedding):
        super().__init__()
        self.embedding = embedding
        self.z_in = nn.Embedding(max_in_degree+1,embedding)
        self.z_out = nn.Embedding(max_out_degree+1,embedding)
    def forward(self,x,in_degree,out_degree):
        _,_,N,d = x.shape
        assert in_degree.shape[0] == N and out_degree.shape[0] == N,"Node dimensions does not match"
        assert d==self.embedding, "Feature embedd dimensions does not match"
        centrality = self.z_in(in_degree)+self.z_out(out_degree)
        x+=centrality
        return x

class SpatialAttensionBias(nn.Module):
    def __init__(self,adj):
        super().__init__()
        self.dist,_ = torch.from_numpy(self.floyd_warshall(adj)).long()
        N = self.dist.shape[0]
        self.attn_bias = np.zeros((N,N),dtype = torch.float)
    def forward(self):
        pass
    def find_path(path, i, j):
        if path[i, j] == 0:
            return [i, j]
        else:
            k = path[i, j]
            return find_path(path, i, k)[:-1] + find_path(path, k, j)
    def floyd_warshall(adj):    
        if os.path.exists('floyd_warshall.npz'):
            mat = np.load("floyd_warshall.npz")
            dist,path=mat['a'],mat['b']
            return dist,path
        
        n = adj.shape[0]
        dist = adj.clone()
        path = torch.zeros_like(adj,dtype=torch.int64)
        dist[dist == 0] = float('inf')
        for i in range(n):
            dist[i][i] = 0
        count = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    count+=1
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                            path[i,j] = k
                    print(f"{count}/{n**3}")
        dist[dist>=510]=510
        np.savez('floyd_warshall',a=dist.numpy(),b=path.numpy())
        return dist,path

class GraphormerEncoder(nn.Module):
    def __init__(self,N:int=307,C:int=1,T:int=12,d:int=6,hno:int=2,layer_no:int=4):
        super().__init__()
        self.N = N
        self.C = C
        self.T = T
        self.d = d
        assert d%hno == 0, "hno should be a perfect divisor of d"
        self.hno = hno
        self.conv = FC(
            input_dims=[C,d],
            units=[d,d],
            activations=[nn.GELU(),None],
            use_bias = False
        )
        self.c_encoder = CentralityEncoder(max_in_degree,max_out_degree,d)
        self.time_embeddings = nn.Embedding(T,d).to(device)
        self.node_embeddings = nn.Embedding(N,d).to(device)
        
        self.input_projection = nn.Linear(d,d).to(device)
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d,nhead=hno,batch_first=True).to(device)
        self.encoder = nn.TransformerEncoder(self.transformer_layer,num_layers = layer_no).to(device)
        
        self.out_projection = nn.Linear(d,C).to(device)
        
        # self.register_buffer("casual_mask",self._generate_casual_mask)
    def _generate_casual_mask(self):
        temp_ids = torch.arange(self.T).repeat_interleave(self.N)
        mask = temp_ids.unsqueeze(0)<temp_ids.unsqueeze(1)
        return mask.float().masked_fill(mask,float('-inf')).to(device)
    
    def forward(self,x):
        B,T,N,C = x.shape
        assert (T,N,C) == (self.T,self.N,self.C),"Dimensions doesn't match"
        x=self.conv(x)
        x=self.c_encoder(x)
        x=x.contiguous().view(B,T*N,self.d)
        temp_ids = torch.arange(T).unsqueeze(0).expand(N,T).reshape(-1).to(device)
        spatial_ids = torch.arange(N).unsqueeze(1).expand(N,T).reshape(-1).to(device)
        
        temp_emb = self.time_embeddings(temp_ids)
        spatial_emb = self.node_embeddings(spatial_ids)
        
        pos_enc = temp_emb+spatial_emb
        
        pos_enc = pos_enc.unsqueeze(0).expand(B,-1,-1)
        
        x=self.input_projection(x)+pos_enc
        
        mask = self._generate_casual_mask()
        encodings = self.encoder(x,mask=mask)
        
        out = self.out_projection(encodings)
        out = out.view(B,T,N,C)
        
        return out

print("Loading Dataset")
traind,_,_,_ = PrepareDataset(pd.read_csv("pems04_flow.csv"),limit=1000)
model = GraphormerEncoder(N=307,C=1,T=12).to(device)
criterion = nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

epochs = 5
print("Training Started")
model.train()
batches = len(traind)
for epoch in range(epochs):
    c_batch =0
    epoch_loss=0.0
    for x,y in traind:
        c_batch+=1
        print(f"Batch {c_batch}/{batches}")
        out = model(x)
        
        loss = criterion(out,y)
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        epoch_loss+=loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/batches:.4f}")
        