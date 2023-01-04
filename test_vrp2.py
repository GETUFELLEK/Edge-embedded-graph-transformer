import torch
import os
import numpy as np
from torch_geometric.data import Data, DataLoader
from creat_vrp import reward1,creat_instance
# from rolloutModel import Model
from gtppo_modelimplicit import Actor_critic
import os
import time
import sys
import torch
from creat_vrp import creat_data,reward,reward1
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import numpy as np
import wandb
n_nodes=101
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
print(device)
def rollout(model, dataset,batch_size,steps):

    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model.act(bat,0,steps,batch_size,True,False)

            cost = reward1(bat.x,cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost
# def rollout(model, dataset,n_nodes):
#     model.eval()
#     def eval_model_bat(bat):
#         with torch.no_grad():
#             cost, _ = model(bat,0,,True)
#             cost = reward1(bat.x,cost.detach(), n_nodes)
#         return cost.cpu()
#     totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
#     return totall_cost

def evaliuate(valid_loder,batch_size,n_node):
    # folder = 'vrp-51-gf'
    folder= 'vrp-{}-gf'
    # n_nodes*2,False,lr,3,hidden_node_dim,1,hidden_edge_dim, epoch,batch_size,laysers,entropy_value,eps_clip,timestep,ppo_epoch
    agent = Actor_critic(3,128,1,64,laysers=4).to(device)
    # agent.load_state_dict(torch.load(''))
    agent.to(device)

    # filepath = os.path.join(folder, '%s' % n_node)
    # filepath = '/home/farid/projects/VRP_RL/RoutingwithGTransformer/vrp-{}-gf/rollout_results/'
    # filepath = '/home/dl-box/RoutingwithGTransformer/Vrp-21-GT2021/rollout_results_2021-10-05 20:36:45.821741/64'

    # filepath = '/home/dl-box/RoutingwithGTransformer/Vrp-21-GT2021/rollout_results_2021-10-10 16:16:56.859804/80'
    # filepath = '/home/dl-box/RoutingwithGTransformer/vrp-51-gf/vrp50results(ppo)2021-10-29 21:59:46.755098/99'
    # filepath ='/home/dl-box/RoutingwithGTransformer/Vrp-101-GT2021/pporesults(100)2021-11-03 09:18:47.353564/90'
    # filepath = '/home/dl-box/RoutingwithGTransformer/vrp-{}-gf/ppo21results2021-11-13 10:33:17.670020/64'
    # =6.18 result with embedding of 512 qnd bachsize 128
    # =6.2111 BRAVO node embedding is 512 and bachsize is 128'
    filepath = '//home/dl-box/RoutingwithGTransformer/vrp-{}-gf/ppo21results2021-12-01 20:10:58.611987/4'
    if os.path.exists(filepath):
        path1 = os.path.join(filepath, 'actor.pt')
        agent.load_state_dict(torch.load(path1, device))
    cost = rollout(agent, valid_loder,batch_size, n_node)
    cost = cost.mean()
    print('Problem:TSP''%s' % n_node,'/ Average distance:',cost.item())

    cost1=cost.min()

    return cost,cost1


def test(n_node):
    datas = []
    # /home/dl-box/RoutingwithGTransformer/test_data/vrp50_test_data.csv
    # / /home/dl-box/RoutingwithGTransformer/test_data/vrp50_capcity.csv
    # /home/dl-box/RoutingwithGTransformer/test_data/vrp50_demand.csv

    if n_node==21 or n_node==51 or n_node==101:
       node_ = np.loadtxt('/home/dl-box/RoutingwithGTransformer/test_data/vrp100_test_data.csv'.format(n_node-1), dtype=np.float, delimiter=',')
       demand_ = np.loadtxt('/home/dl-box/RoutingwithGTransformer/test_data/vrp100_demand.csv'.format(n_node-1), dtype=np.float, delimiter=',')
       capcity_ = np.loadtxt('/home/dl-box/RoutingwithGTransformer/test_data/vrp100_capcity.csv'.format(n_node-1), dtype=np.float, delimiter=',')
       # node_= np.loadtxt('/home/dl-box/RoutingwithGTransformer/CVRPLIB/coordinatesAn32k5.csv'.format(n_node-1),dtype=np.float, delimiter=',')
       # capcity_=np.loadtxt('/home/dl-box/RoutingwithGTransformer/CVRPLIB/capacity.csv',dtype=np.float, delimiter=',')
       # demand_ = np.loadtxt('/home/dl-box/RoutingwithGTransformer/CVRPLIB/An32k5demandnormalized.csv', dtype=np.float,delimiter=',')
        # capcity = /home/dl-box/RoutingwithGTransformer/test_data/vrp100_capcity.csv
       batch_size=128
    else:
      print('Please enter 21, 51 or 101')
      return
    node_ = node_.reshape(-1, n_node, 2)
    # Calculate the distance matrix
    def c_dist(x1,x2):
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    #edges = torch.zeros(n_nodes,n_nodes)

    data_size = node_.shape[0]

    edges = np.zeros((data_size, n_node, n_node, 1))
    for k, data in enumerate(node_):
        for i, (x1, y1) in enumerate(data):
            for j, (x2, y2) in enumerate(data):
                d = c_dist((x1, y1), (x2, y2))
                edges[k][i][j][0] = d
    edges_ = edges.reshape(data_size, -1, 1)

    edges_index = []
    for i in range(n_node):
        for j in range(n_node):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)
    for i in range(data_size):

        data = Data(x=torch.from_numpy(node_[i]).float(), edge_index=edges_index, edge_attr=torch.from_numpy(edges_[i]).float(),
                    demand=torch.tensor(demand_[i]).unsqueeze(-1).float(),
                    capcity=torch.tensor(capcity_[i]).unsqueeze(-1).float())
        datas.append(data)

    print('Data created')
    dl = DataLoader(datas, batch_size=batch_size)
    evaliuate(dl,batch_size,n_node)

test(101)