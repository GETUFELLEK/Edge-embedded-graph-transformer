import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data,DataLoader
from tqdm import tqdm
import os
import pickle


def creat_instance_npy(n_nodes):
    
    
    CAPACITIES = {
        10: 2.,
        20: 3.,
        50: 4.,
        100: 5.
    }

    demand = np.random.randint(1, 10, size=(n_nodes-1)) # Demand, uniform integer 1 ... 9
    demand = np.array(demand)/10
    demand = np.insert(demand,0,0.)
    capcity = CAPACITIES[n_nodes-1]
    return demand,capcity#demand(num,node) capcity(num)


def creat_data_npy(n_nodes,file_list ,batch_size=32, samples=1000):
    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0,dim1=1)

    datas = []
    
    edges = np.load(file_list[0])
    data_map = np.load(file_list[1])
    
    for i in range(samples):
        node = data_map[i]
        edge = edges[i].reshape(-1, 1)
        demand, capcity = creat_instance_npy(n_nodes)
        data = Data(x=torch.from_numpy(node).float(), edge_index=edges_index,edge_attr=torch.from_numpy(edge).float(),
                    demand=torch.tensor(demand).unsqueeze(-1).float(),capcity=torch.tensor(capcity).unsqueeze(-1).float())
        datas.append(data)
    #print(datas)
    dl = DataLoader(datas, batch_size=batch_size)
    return dl

if __name__ == "__main__":
    
    node_size_list = [21,51,101]
    city_list = ['hibikino','cairo','shinjuku', 'adisababa', 'namba', 'hakata']
    
    for node_size in node_size_list:
        for city in city_list:
            print(node_size,city)
            map_norm = city+'_'+str(node_size)+'_map_norm.npy'
            map_loc_norm = city+'_'+str(node_size)+'_map_loc_norm.npy'
            dl = creat_data_npy(node_size,[map_norm,map_loc_norm] ,batch_size=32)
            with open(city+'_'+str(node_size)+'_dataloaders.pkl', 'wb') as file:
              pickle.dump(dl, file)