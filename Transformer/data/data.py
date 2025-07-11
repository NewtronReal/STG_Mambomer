import torch
import numpy as np
import torch.utils.data as utils
import pandas as pd


def PrepareDataset(speed_matrix, BATCH_SIZE=30, seq_len=12, pred_len=12, train_propotion=0.7, valid_propotion=0.1,limit=100000000,win_size=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    speed_sequences, speed_labels = torch.Tensor(np.asarray(speed_sequences)), torch.Tensor(np.asarray(speed_labels))
    # Reshape labels to have the same second dimension as the sequences
    speed_sequences
    
    ##Sliding windows
    speed_sequences = speed_sequences.contiguous().permute(0,2,1).unfold(dimension=-1,size=win_size, step=1).contiguous().permute(0,2,3,1)
    speed_labels = speed_labels.contiguous().permute(0,2,1).unfold(dimension=-1,size=win_size, step=1).contiguous().permute(0,2,3,1)
    print(speed_labels.shape,speed_sequences.shape)
    # shuffle & split the dataset to training and testing sets
    sample_size = speed_sequences.shape[0]
    # index = np.arange(sample_size, dtype=int)
    # np.random.shuffle(index)

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

def get_graph_info(adj_path:str = "pems04_adj.npy"):
    adj = torch.tensor(np.load(adj_path))
    in_degree = (adj!=0).sum(dim=0)
    out_degree = (adj!=0).sum(dim=1)
    max_in_degree = max(in_degree)
    max_out_degree = max(out_degree)
    return adj,in_degree,out_degree,max_in_degree,max_out_degree

""" 
Loads Dataset with B,T,m,N,C will be replacing PrepareDataset in future modified version that of STG-Mamba
"""

def load_sliding_windows(datapath="pems04_flow.csv",
                   bsize:int=4,seql:int=5,preds:int=5,
                   train_prop:float=.7,valid_prop:float=.1,limit:int=20,
                   ws=3):
    data = pd.read_csv(datapath)[:limit]
    ## min-max normalization
    max_f = data.max().max()
    min_f = data.min().min()
    data = (data -min_f)/(max_f-min_f)
    
    windows,labels = [],[]

    for i in range(data.shape[0]-ws):
        windows.append(data.iloc[i:i+ws].values)
        labels.append(data.iloc[i+1:i+ws+1].values)
    
    windows,labels = torch.tensor(np.asarray(windows)),torch.tensor(np.asarray(labels))
    
    sequences,seq_labels = [],[]
    
    for i in range(0,windows,seql):
        sequences.append(windows[i:i+seql])
        seq_labels.append(labels[i:i+seql])
    
    sequences,seq_labels = torch.tensor(np.asarray(sequences)),torch.tensor(np.asarray(seq_labels))
    
    return windows,labels