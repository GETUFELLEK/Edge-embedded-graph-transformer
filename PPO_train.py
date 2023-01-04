
import os
import time
import torch
# from import Agentppo,Memory
from gtppo_modelimplicit import Agentppo,Memory
from creat_vrp import creat_data,multi_threaded_created_data,creat_data2, reward,reward1
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import numpy as np
import datetime
import pickle
import wandb
# wandb.init()
#         config = wandb.config
#         config.learning_rate = 1e-4
#         config.epochs = 100
#         config.batch_size = 512
use_wandb = True
device = torch.device('cuda:0')
                      # if torch.cuda.is_available() else 'cpu')
print(device)
#device = torch.device('cpu')
n_nodes =21
# n_heads=4
def rollout(model, dataset,batch_size,steps):

    model.eval()
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model.act(bat,0,steps,batch_size,True,False)

            cost = reward1(bat.x,cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost
class TrainPPO:
    def __init__(self,steps,greedy,lr,input_node_dim,hidden_node_dim,input_edge_dim,hidden_edge_dim,epoch=40,batch_size=32,laysers=3,entropy_value=0.01,eps_clip=0.2,timestep=4,ppo_epoch=2):
        # self.n_heads=n_heads
        self.steps = steps
        self.greedy = greedy
        self.batch_size = batch_size
        self.update_timestep = timestep
        self.epoch =epoch
        self.memory = Memory()
        self.agent =     Agentppo(steps,greedy,lr,input_node_dim,hidden_node_dim,input_edge_dim,hidden_edge_dim,ppo_epoch,batch_size,laysers,entropy_value,eps_clip)


    def run_train(self,data_loader,batch_size,valid_loder,gurobi_baseline=6.10):
        memory = Memory()
        self.agent.old_polic.to(device)
        #initWeights(self.agent.old_polic)
        #initWeights(self.agent.policy)
        folder= 'vrp-{}-gf'
        # folder = 'vrp-{}-gf'.format(n_nodes)
        filename = 'ppo21results'+str(datetime.datetime.now())
        filepath = os.path.join(folder, filename)

        '''path = os.path.join(filepath,'%s' % 3)
        if os.path.exists(path):
            path1 = os.path.join(path, 'actor.pt')
            self.agent.old_polic.load_state_dict(torch.load(path1, device))'''

        costs = []
        for i in range(self.epoch):
            print('old_epoch:', i, '***************************************')
            self.agent.old_polic.train()
            times, losses, rewards2, critic_rewards = [], [], [], []
            epoch_start = time.time()
            start = epoch_start
            for batch_idx, batch in enumerate(data_loader):

                x,attr,capcity,demand = batch.x,batch.edge_attr,batch.capcity,batch.demand
                #print(x.size(),index.size(),attr.size())
                x,attr,capcity,demand = x.view(batch_size,n_nodes,2),attr.view(batch_size,n_nodes*n_nodes,1),capcity.view(batch_size,1),demand.view(batch_size,n_nodes,1)
                batch = batch.to(device)
                actions, log_p = self.agent.old_polic.act(batch,0,self.steps,batch_size,self.greedy,False)

                rewards = reward1(batch.x, actions, n_nodes)

                actions = actions.to(torch.device('cpu')).detach()
                log_p = log_p.to(torch.device('cpu')).detach()
                rewards = rewards.to(torch.device('cpu')).detach()

                #print(actions.size(),log_p.size(),entropy.size())

                for i_batch in range(self.batch_size):
                    memory.input_x.append(x[i_batch])
                    #memory.input_index.append(index[i_batch])
                    memory.input_attr.append(attr[i_batch])
                    memory.actions.append(actions[i_batch])
                    memory.log_probs.append(log_p[i_batch])
                    memory.rewards.append(rewards[i_batch])
                    memory.capcity.append(capcity[i_batch])
                    memory.demand.append(demand[i_batch])
                if (batch_idx+1)%self.update_timestep == 0:
                    self.agent.update(memory,i)
                    memory.def_memory()
                rewards2.append(torch.mean(rewards.detach()).item())
                time_Space = 100
                if (batch_idx+1) % time_Space == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end
                    mean_reward = np.mean(rewards2[-time_Space:])
                    print('  Batch %d/%d, reward: %2.3f,took: %2.4fs' %
                          (batch_idx, len(data_loader), mean_reward,
                           times[-1]))
            cost = rollout(self.agent.policy, valid_loder, batch_size, self.steps)
            cost = cost.mean()
            distance= cost.mean()-gurobi_baseline
            Optimality_Gap=(distance/gurobi_baseline)*100
            costs.append(cost.item())
            print('Problem:TSP''%s' % n_nodes,'/ Average distance:',cost.item())
            wandb.log({"Opt.Gap": Optimality_Gap})
            wandb.log({"average_distance": cost.mean()})

            print(costs)
            epoch_dir = os.path.join(filepath, '%s' % i)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(self.agent.old_polic.state_dict(), save_path)
def train():
    class RunBuilder():
        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    params = OrderedDict(
        lr=[1e-4],
        hidden_node_dim=[128],
        hidden_edge_dim=[64],
        epoch = [100],
        batch_size=[512],
        laysers=[4],
        entropy_value=[0.01],
        eps_clip=[0.2],
        timestep=[1],
        ppo_epoch=[3],
        data_size=[768000],
        valid_size=[10000]
    )
    runs = RunBuilder.get_runs(params)

    for lr, hidden_node_dim, hidden_edge_dim, epoch,batch_size,laysers,entropy_value,eps_clip,timestep,ppo_epoch ,data_size,valid_size in runs:
        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'laysers', 'epoch,batch_size',
              'entropy_value', 'eps_clip', 'timestep:','data_size','valid_size', lr, hidden_node_dim,
              hidden_edge_dim, epoch, batch_size, laysers, entropy_value, eps_clip, timestep,data_size,valid_size)
        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'laysers:', lr, batch_size, hidden_node_dim,
              hidden_edge_dim, laysers)
        data_pkl_filename = 'data_' + str(n_nodes) + '_' + str(data_size) + '_' + str(batch_size) + '.pkl'
        valid_pkl_filename = 'valid_' + str(n_nodes) + '_' + str(valid_size) + '_' + str(batch_size) + '.pkl'

        if os.path.exists(data_pkl_filename):
            with open(data_pkl_filename, mode='rb') as f:
                data_loder = pickle.load(f)
        else:
            data_loder = creat_data(n_nodes, data_size, batch_size)
            with open(data_pkl_filename, mode='wb') as f:
                pickle.dump(data_loder, f)

        if os.path.exists(valid_pkl_filename):
            with open(valid_pkl_filename, mode='rb') as f:
                valid_loder = pickle.load(f)
        else:
            valid_loder = creat_data(n_nodes, valid_size, batch_size)
            with open(valid_pkl_filename, mode='wb') as f:
                pickle.dump(valid_loder, f)
        # data_loder = multi_threaded_created_data(n_nodes,data_size,batch_size)
        # valid_loder = multi_threaded_created_data(n_nodes, valid_size, batch_size)
        print('DATA CREATED/Problem size:', n_nodes)
        trainppo = TrainPPO(n_nodes*2,False,lr,3,hidden_node_dim,1,hidden_edge_dim, epoch,batch_size,laysers,entropy_value,eps_clip,timestep,ppo_epoch)
        trainppo.run_train(data_loder,batch_size,valid_loder)
# with torch.cuda.amp.autocast(enabled=False):
# wandb.log({
#         "Epoch": epoch,
# /home/dl-box/RoutingwithGTransformer/PPO_train.py
if use_wandb:

        wandb.init(project="optimalitygapvrp21", entity="getutadesse",group="experiment_1",job_type="ours")
        config = wandb.config
        config.learning_rate = 1e-4
        config.epochs = 100
        config.batch_size = 512

train()