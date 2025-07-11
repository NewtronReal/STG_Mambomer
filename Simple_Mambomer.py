import numpy as np
import torch
import scipy.sparse as sp

adj = np.load("T-Graphormer/ST-Mambomer/Mamba/PEMS04/pems04_adj.npy")
adj2 = 1/torch.tensor(sp.load_npz("T-Graphormer/src/data/traffic/pems-bay/adj.npz").toarray(),dtype=torch.float)
adj = torch.Tensor(adj)

def find_path(path,i,j,nodes:list=[]):
    nodes.append(int(i))
    if path[i,j]==0:
        nodes.append(j)
        return nodes
    else:
        return find_path(path,path[i,j],j,nodes)

def floyd_warshall(adj,url="floydwarshall"):
    import os
    
    if os.path.exists(url):
        mat = np.load(url)
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
    np.savez(url.replace(".npz",""),a=dist.numpy(),b=path.numpy())
    return dist,path


dist,path = floyd_warshall(adj2,url="floydwarshall2.npz")

print(dist[1,42]+dist[42,2],dist[1,2])
print(np.max(dist,axis=0))
print(find_path(path,1,2))