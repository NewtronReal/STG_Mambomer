import torch
import numpy as np
import os

def find_path(path, i, j):
    if path[i, j] == 0:
        return [i, j]
    else:
        k = path[i, j]
        return find_path(path, i, k)[:-1] + find_path(path, k, j)

def floyd_warshall(adj,clip:int=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filename = 'floyd_warshall_clip'+str(clip)  
    if os.path.exists(filename+'.npz'):
        mat = np.load(filename+'.npz')
        dist,path=mat['a'],mat['b']
        return torch.tensor(dist).to(device),torch.tensor(path).to(device)
    
    n = adj.shape[0]
    dist = adj.clone()
    path = torch.zeros_like(adj,dtype=torch.int64).to(device)
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
    dist[dist>=clip]=clip
    np.savez(filename,a=dist.numpy(),b=path.numpy())
    return torch.tensor(dist).to(device),torch.tensor(path).to(device)