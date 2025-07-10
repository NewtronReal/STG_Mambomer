import torch
import torch.nn as nn
import pandas as pd
from data.data import PrepareDataset
from modules.GraphFeatures import GraphFeatures
from modules.GraphormerEncoder import GraphormerEncoder

""" 
Training script with cosannlealing adamw and mseloss
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Add adjacency matrix for encoding graph info

graph = GraphFeatures()

print("Loading Dataset")
traind,_,_,_ = PrepareDataset(pd.read_csv("pems04_flow.csv"),limit=1000)
model = GraphormerEncoder(N=307,C=1,T=12,graph=graph).to(device)
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
        