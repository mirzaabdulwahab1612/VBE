from audioop import bias
import imp
from operator import index, ne
from os import stat
import numpy as np
import torch
import torch.optim as optim
from scipy.special import softmax
import matplotlib.pyplot as plt

from ..utils.agent_utils import *
from utils.replay_buffer import SimpleReplayBuffer
from utils.PlotDeepSea import PlotDeepSea
import copy
import math
import os

class OPI():

    def __init__(self,
                input_size,
                output_size,
                gamma,
                learning_rate,
                np_random = np.random.RandomState(0),
                target_update_freq = 10000,
                update_network_freq = 4,
                num_updates = 1,
                mini_batch_size = 64,
                device = torch.device("cpu"),
                num_rvecs = 1,
                p = 1,
                target_polic_type = "greedy",
                rvfs_update_all = False,
                model_save_path = None,
                ):

        # Replay buffer
        self.replay_buffer_train = SimpleReplayBuffer(np_random, max_buffer_length=100000)

        self.num_action = output_size
        self.gamma = gamma
        self.alpha = learning_rate
        self.np_random = np_random
        # update frequencies
        self.target_update_freq = target_update_freq
        self.update_network_freq = update_network_freq
        self.num_updates = num_updates
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.num_rvecs = num_rvecs
        
        self.wvec = Net(input_size, self.num_action).to(self.device)
        self.target_wvec = Net(input_size, self.num_action).to(self.device)

        self.rand_target = []
        self.rand_pred = []
        self.target_rand_pred = []
        for i in range(self.num_rvecs):
            # change: RQFs have the same architecture as the Q-function updating CNN as well
            # self.rand_target.append(Net(input_size, self.num_action, rqf_target=True).to(self.device))
            # self.rand_pred.append(Net(input_size, self.num_action).to(self.device))
            # self.target_rand_pred.append(Net(input_size, self.num_action).to(self.device))

            # change: RQFs have the same representation network (weights), only updating the linear layer
            self.rand_target.append(RVFs(input_size, self.num_action).to(self.device))
            self.rand_pred.append(RVFs(input_size, self.num_action).to(self.device))
            self.target_rand_pred.append(RVFs(input_size, self.num_action).to(self.device))

        # change: RQFs have the same representation network (weights), only updating the linear layer
        for i in range(self.num_rvecs):
            self.rand_pred[i].representation.load_state_dict(self.rand_target[i].representation.state_dict())

        # Target networks
        self.target_wvec.load_state_dict(self.wvec.state_dict())
        for i in range(self.num_rvecs):
            self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())
        
        # parameters for the bound
        self.p = p
        self.target_polic_type = target_polic_type
        self.rvfs_update_all = rvfs_update_all

        # For storing current state-action-features
        self.time_step = 0
        self.episode_num = 0
        self.start_training = False

        net_params = list(self.wvec.parameters())
        for i in range(self.num_rvecs):
            # change: RQFs have the same architecture as the Q-function updating CNN as well
            # net_params += list(self.rand_pred[i].parameters())

            # change: RQFs have the same representation network (weights), only updating the linear layer
            net_params += list(self.rand_pred[i].net.parameters())
        # Optimizer
        self.optimizer = optim.Adam(net_params, lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        # new change: updating after 1000 random steps
        self.start_training_step = 1000
        
        self.path = model_save_path
        self.model_save_frequency = 3000000
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except:
                assert (os.path.exists(self.path))

    def start(self, state):
        next_act, _, _, _ = self.optimistic_action(state, target_action=False)
        self.current_state = copy.deepcopy(state)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, state, reward, terminal):
        # Storing data in the replay_buffer
        self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(state), copy.deepcopy(terminal))
        next_act, val, uncertainty_bonus, value = self.optimistic_action(state, target_action=False)

        # Control update frequency
        if(self.time_step % self.target_update_freq == 0 and self.time_step > 0):
            self.target_wvec.load_state_dict(self.wvec.state_dict())
            for i in range(self.num_rvecs):
                self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())

        # new change: updating after 1000 random steps
        if((self.time_step > self.start_training_step) and (self.replay_buffer_train.get_buffer_size() > self.mini_batch_size)):
            self.start_training = True
            if self.time_step % self.update_network_freq == 0:
                self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(state)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1

        # new change: save the model
        if self.time_step % self.model_save_frequency == 0:
            self.save_policy(self.path)

        return next_act, val, uncertainty_bonus, value

    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            if self.target_polic_type == "optimistic" or self.target_polic_type == "optimistic-uniform":
                next_actions, _, _, _ = self.optimistic_action(next_states, target_action=True)
            elif self.target_polic_type == "greedy" or self.target_polic_type == "greedy-greedy" or self.target_polic_type == "greedy-uniform":
                next_actions = np.argmax(self.wvec.forward(next_states_t).detach().cpu(), axis=-1)
            else:
                exit("Invalid value target policy type")
            next_values = self.target_wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            if self.rvfs_update_all:
                # Updating all RVFs
                rvfs_to_update = np.arange(self.num_rvecs)
            else:
                # Updating a random RVF
                rvfs_to_update = self.np_random.randint(0, self.num_rvecs, 1)

            if self.target_polic_type == "optimistic" or self.target_polic_type == "greedy":
                next_actions = [next_actions] * self.num_rvecs
            elif self.target_polic_type == "optimistic-uniform" or self.target_polic_type == "greedy-uniform":
                next_actions = [self.np_random.randint(0, self.num_action, mini_batch_size)] * self.num_rvecs
            elif self.target_polic_type == "greedy-greedy":
                rand_p = torch.stack([self.rand_pred[id].forward(next_states_t).detach() for id in range(self.num_rvecs)])
                rand_t = torch.stack([self.rand_target[id].forward(next_states_t).detach() for id in range(self.num_rvecs)])
                iunc = torch.abs(rand_p - rand_t)
                iunc = torch.max(iunc, axis=0)[0]

                next_actions = torch.argmax(iunc, axis=-1)
                next_actions = [next_actions] * self.num_rvecs
            else:
                exit("Invalid value target policy type")

            for j in rvfs_to_update:

                next_values_target = self.rand_target[j].forward(next_states_t).detach()
                next_q_target = next_values_target[np.arange(mini_batch_size), next_actions[j]]
                current_values_target = self.rand_target[j].forward(states_t).detach()
                current_q_target = current_values_target[np.arange(mini_batch_size), actions]
                reward_u = current_q_target - (self.gamma*terminals_*next_q_target)

                next_values_pred = self.target_rand_pred[j].forward(next_states_t).detach()
                next_q_pred = next_values_pred[np.arange(mini_batch_size), next_actions[j]]
                current_values_pred = self.rand_pred[j].forward(states_t)
                current_q_pred = current_values_pred[np.arange(mini_batch_size), actions]
                target_q_pred = reward_u + self.gamma*terminals_*next_q_pred
                loss += self.loss_fn(current_q_pred, target_q_pred)

            # updating weights backpropagation        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def eval_step(self, state, reward=None, terminal=None, step=0):
        with torch.no_grad():
            target_action=False
            hidden_state = torch.Tensor().to(self.device)
            val = self.wvec.forward(hidden_state)
            value_new = val.data.cpu().numpy()
            act = get_greedy_action(value_new, self.np_random, target_action)
            return act
    
    def optimistic_action(self, state, target_action=False):
        with torch.no_grad():

            val = torch.zeros((self.num_action)).data.cpu().numpy()
            uncertainty_bonus = torch.zeros((self.num_action)).data.cpu().numpy()
            if not target_action:
                hidden_state = torch.Tensor(np.array([state])).to(self.device)
            else:
                hidden_state = torch.Tensor(state).to(self.device)

            if self.start_training:
                val = self.wvec.forward(hidden_state)
                rand_p = torch.stack([self.rand_pred[id].forward(hidden_state) for id in range(self.num_rvecs)])
                rand_t = torch.stack([self.rand_target[id].forward(hidden_state) for id in range(self.num_rvecs)])
                iunc = torch.abs(rand_p - rand_t)
                iunc = torch.max(iunc, axis=0)[0]

                uncertainty_bonus = self.p * (iunc)
                value_new = (val + uncertainty_bonus).data.cpu().numpy()
                if not target_action:
                    val = val[0]
                    uncertainty_bonus = uncertainty_bonus[0]
                    value_new = value_new[0]
            else:
                value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                
            act = get_greedy_action(value_new, self.np_random, target_action)

        return act, val, uncertainty_bonus, value_new

    def save_policy(self, path):
        torch.save(self.wvec.state_dict(), str(path+"wvec_weights_at_step_"+str(self.time_step)))
        torch.save(self.target_wvec.state_dict(), str(path+"target_wvec_weights_at_step_"+str(self.time_step)))
        torch.save(self.optimizer.state_dict(), str(path + "_optimizer_state_at_step_"+str(self.time_step)))

        for i in range(self.num_rvecs):
            torch.save(self.rand_pred[i].state_dict(), str(path+"rand_pred_"+str(i)+"_weights_at_step_"+str(self.time_step)))
            torch.save(self.rand_target[i].state_dict(), str(path+"rand_target_"+str(i)+"_weights_at_step_"+str(self.time_step)))
            torch.save(self.target_rand_pred[i].state_dict(), str(path+"target_rand_pred_"+str(i)+"_weights_at_step_"+str(self.time_step)))

def init(params):
    return OPI(params)

def get_params():
    return ["optimizer_type", "target_polic_type", "rvfs_update_all", "num_rvecs", "alpha", "p"]

class Net(torch.nn.Module):
    def __init__(self, inputSize, outputSize, rqf_target=False):
        super(Net, self).__init__()

        self.representation = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                7 * 7 * 64,
                256),
            torch.nn.ReLU(),
            torch.nn.Linear(
                256,
                448),
            torch.nn.ReLU()
        )

        self.net = torch.nn.Sequential(
            torch.nn.Linear(448, 448),
            torch.nn.ReLU(),
            torch.nn.Linear(448, outputSize)
        )

        # new change: initializing the weights of the linear layers with 1 scale for target RQFs
        lin_scale = 0.01
        if(rqf_target):
            lin_scale = 1

        for p in self.modules():
            if isinstance(p, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, torch.nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # new change: initializing the weights of the linear layers with 1 scale for target RQFs
        for i in range(len(self.net)):
            if type(self.net[i]) == torch.nn.Linear:
                torch.nn.init.orthogonal_(self.net[i].weight, lin_scale)
                self.net[i].bias.data.zero_()

    def forward(self, state):
        state = self.representation(state)
        return self.net(state)
    
class RVFs(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(RVFs, self).__init__()
        
        self.representation = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                7 * 7 * 64,
                256),
            torch.nn.ReLU(),
            torch.nn.Linear(
                256,
                448),
            torch.nn.ReLU(),
        )

        self.net = torch.nn.Sequential(
            torch.nn.Linear(448, outputSize, bias=True))
        
        for p in self.modules():
            if isinstance(p, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, torch.nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for i in range(len(self.net)):
            if type(self.net[i]) == torch.nn.Linear:
                torch.nn.init.orthogonal_(self.net[i].weight, 0.01)
                self.net[i].bias.data.zero_()
    
    def forward(self, x):
        x = self.representation(x).detach()
        return self.net(x)
