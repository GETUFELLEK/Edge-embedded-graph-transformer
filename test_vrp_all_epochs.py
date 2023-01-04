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
import os
use_wandb = True
# from VRP_Actor import Model
n_nodes=21
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')
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


def evaliuate(valid_loder,n_node):
    folder= 'ppo21results2021-11-13 10'
    batch_size=128

    epochs = os.listdir(folder)
    costs = []
    costs1 = []
    for epoch in epochs:
        agent = Actor_critic(3, 512, 1, 64, laysers=4).to(device)

        agent.to(device)


    #     filepath = os.path.join(folder, '%s' % n_node)
        filepath = os.path.join(folder,epoch)
        # print(filepath)

        if os.path.exists(filepath):
            path1 = os.path.join(filepath, 'actor.pt')
            agent.load_state_dict(torch.load(path1, device))
        cost = rollout(agent, valid_loder,batch_size, n_node)
        cost = cost.mean()
        costs.append(cost)
        print('Problem:TSP''%s' % n_node,'/ Average distance:',cost.item())
        gurobi_baseline= 6.1
        cost1=cost.min()
        costs1.append(cost1)
        distance_ = cost.item() - gurobi_baseline
        Optimality_Gap = (distance_ / gurobi_baseline) * 100


        wandb.log({'epochs':epoch,"Opt.Gap": Optimality_Gap,})
        # wandb.log({"average_distance": cost.mean()})
        # wandb.plot.op_curve()
        # wandb.log({"average distance vs Optimality gap": wandb.plot.op_curve(epochs, Optimality_Gap)})
        # wandb.log({"average distance versus epochs": (epoch, Optimality_Gap)})


    return costs,costs1
def test(n_node):
    datas = []

    if n_node==21 or n_node==51 or n_node==101:
        if n_node == 21 or n_node == 51 or n_node == 101:
            node_ = np.loadtxt('/home/dl-box/RoutingwithGTransformer/test_data/vrp20_test_data.csv'.format(n_node - 1),
                               dtype=np.float, delimiter=',')
            demand_ = np.loadtxt('/home/dl-box/RoutingwithGTransformer/test_data/vrp20_demand.csv'.format(n_node - 1),
                                 dtype=np.float, delimiter=',')
            capcity_ = np.loadtxt(
                '/home/dl-box/RoutingwithGTransformer/test_data/vrp20_capcity.csv'.format(n_node - 1), dtype=np.float,
                delimiter=',')
            # node_= np.loadtxt('/home/dl-box/RoutingwithGTransformer/CVRPLIB/coordinatesAn32k5.csv'.format(n_node-1),dtype=np.float, delimiter=',')
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
    evaliuate(dl,n_node)
if use_wandb:

        wandb.init(project="optimality gap", entity="getutadesse",group="experiment_1",job_type="ours")
        config = wandb.config
        config.learning_rate = 1e-4
        config.epochs = 100
        config.batch_size = 128
test(21)