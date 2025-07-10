import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch

speed_matrix=pd.read_csv("Mamba/PEMS04/pems04_flow.csv")

BATCH_SIZE=4

seq_len =3
pred_len =3

train_propotion = .7
valid_propotion=.1


max_speed = speed_matrix.max().max()
min_speed = speed_matrix.min().min()
speed_sequences, speed_labels = [],[]
speed_matrix = (speed_matrix - min_speed)/(max_speed-min_speed)

for i in range(speed_matrix.shape[0]-seq_len-pred_len):
    speed_sequences.append(speed_matrix.iloc[i:i+seq_len].values)
    speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
speed_sequences,speed_labels = np.asarray(speed_sequences),np.asarray(speed_labels)

speed_labels = speed_labels.reshape(speed_labels.shape[0],seq_len,-1)