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
import pickle

class SARSA():

    def __init__(self, params):

        # Replay buffer
        self.replay_buffer_train = SimpleReplayBuffer(params.np_random)

        self.model_save_path = params.basic.path+"/"+str(params.repeat)+"_"
        self.model_save_freq = int(params.agent_params.model_save_freq)
        self.num_action = params.environment.num_action
        self.obs_dim = params.environment.obs_dim
        self.feature_dim = params.feature_constructor.feature_dim
        self.gamma = params.agent_params.gamma
        self.alpha = params.agent_params.alpha
        self.print_mode = params.agent_params.print_mode
        self.print_frequency = params.agent_params.print_frequency
        self.feature_constructor = params.feature_constructor
        self.np_random = params.np_random
        self.logger = params.logger

        # update frequencies
        self.update_freq_policy = params.agent_params.update_freq_policy
        self.num_updates = params.agent_params.num_updates
        self.mini_batch_size = params.agent_params.mini_batch_size
        self.mem_size = self.feature_dim*self.num_action
        self.features = np.zeros(self.feature_dim)

        self.device = params.device
        self.num_rvecs = params.agent_params.num_rvecs
        
        # Linear function approximator
        self.wvec = LinearNet(self.feature_dim, self.num_action, bias=False)
        self.target_wvec = LinearNet(self.feature_dim, self.num_action, bias=False)

        self.rand_target = []
        self.rand_pred = []
        self.target_rand_pred = []
        for i in range(self.num_rvecs):
            # Linear function approximator
            self.rand_target.append(LinearNet(self.feature_dim, self.num_action, bias=False))
            self.rand_pred.append(LinearNet(self.feature_dim, self.num_action, bias=False))
            self.target_rand_pred.append(LinearNet(self.feature_dim, self.num_action, bias=False))

        # Target networks
        for i in range(self.num_rvecs):
            self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())
        
        # parameters for the bound
        self.p = params.agent_params.p
        self.c = params.agent_params.p
        self.action_type = params.agent_params.action_type
        self.rvfs_update_all = params.agent_params.rvfs_update_all

        # For storing current state-action-features
        self.current_state = np.zeros(self.mem_size)
        self.current_action = None
        self.time_step = 0
        self.episode_num = 0
        self.start_training = False

        net_params = list()
        for i in range(self.num_rvecs):
            net_params += list(self.rand_pred[i].parameters())
        # Optimizer
        self.optimizer_type = params.agent_params.optimizer_type
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(net_params, lr=self.alpha)
        elif self.optimizer_type == "adam_without_mom":
            self.optimizer = optim.Adam(net_params, lr=self.alpha, betas=(0, 0.999))
        elif self.optimizer_type == "rmsprop":
            self.optimizer = optim.rmsprop(net_params, lr=self.alpha)
        elif self.optimizer_type == "rmsprop_with_mom":
            self.optimizer = optim.rmsprop(net_params, lr=self.alpha, momentum=0.9)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(net_params, lr=self.alpha)
        elif self.optimizer_type == "sgd_with_mom":
            self.optimizer = optim.SGD(net_params, lr=self.alpha, momentum=0.9)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        self.unique_states = dict()
        self.first_visits = dict()
        self.num_unique_states = []
        
    def start(self, observation):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        self.unique_states[tuple(observation)] = self.unique_states.get(tuple(observation), 0) + 1
        self.first_visits[tuple(observation)] = self.first_visits.get(tuple(observation), self.time_step)
        self.num_unique_states.append(len(self.unique_states))
        next_act = self.optimistic_action(self.features, self.rand_pred, target_action=False)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, observation, reward, terminal):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        self.unique_states[tuple(observation)] = self.unique_states.get(tuple(observation), 0) + 1
        self.first_visits[tuple(observation)] = self.first_visits.get(tuple(observation), self.time_step)
        self.num_unique_states.append(len(self.unique_states))
        # Storing data in the replay_buffer
        self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))
        next_act = self.optimistic_action(self.features, self.rand_pred, target_action=False, print_flag=True)

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            for i in range(self.num_rvecs):
                self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1

        if(self.time_step%self.model_save_freq == 0):
            self.save_stats(self.model_save_path)

        return next_act
    
    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            loss = 0
            if self.rvfs_update_all:
                # Updating all RVFs
                rvfs_to_update = np.arange(self.num_rvecs)
            else:
                # Updating a random RVF
                rvfs_to_update = self.np_random.randint(0, self.num_rvecs, 1)
            
            # Regular target policy for RVFs
            next_actions = self.optimistic_action(next_states, self.rand_pred, target_action=True)
            next_actions = [next_actions] * self.num_rvecs
            
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
            get_features(state, self.feature_constructor, self.features, self.features)
            hidden_state = torch.Tensor(copy.deepcopy(self.features)).to(self.device)
            val = self.wvec.forward(hidden_state)
            value_new = val.data.cpu().numpy()
            act = get_greedy_action(value_new, self.np_random, target_action)
            return act
    
    def optimistic_action(self, state, rand_pred=None, target_action=False, print_flag=False):
        with torch.no_grad():
            hidden_state = torch.Tensor(state).to(self.device)
            if self.start_training:

                rand_p = torch.stack([rand_pred[id].forward(hidden_state) for id in range(self.num_rvecs)])
                rand_t = torch.stack([self.rand_target[id].forward(hidden_state) for id in range(self.num_rvecs)])
                iunc = torch.abs(rand_p - rand_t)

                if (self.action_type == "max"):
                    # take max of the iunc for each action
                    iunc = torch.max(iunc, axis=0)[0]
                elif(self.action_type == "mean"):
                    # take mean of the iunc for each action
                    iunc = torch.mean(iunc, axis=0)
                else:
                    exit("Invalid action type")

                uncertainty_bonus = self.c * (iunc)
                # value_new = (val + uncertainty_bonus).data.cpu().numpy()
                value_new = uncertainty_bonus.data.cpu().numpy()
                # if(not target_action):
                #     if(np.argmax(state) == 0):
                #         print(f"time: {self.time_step//30} ; State: {np.argmax(state)} Number of unique states: {len(self.unique_states)} , first_visits: {self.first_visits[tuple([np.argmax(state)])]}")
                #         print(f"val: {val}, iunc: {iunc}, uncertainty_bonus: {uncertainty_bonus} value_new: {value_new}")
            else:
                value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                
            act = get_greedy_action(value_new, self.np_random, target_action)

        return act

    def save_policy(self, path):
        torch.save(self.wvec.state_dict(), str(path+"weights_at_step_"+str(self.time_step)))

    def save_stats(self, path):
        with open(str(path+"unique_states"), 'wb') as f:
            pickle.dump(self.unique_states, f)
        
        with open(str(path+"first_visits"), 'wb') as f:
            pickle.dump(self.first_visits, f)

        with open(str(path+"num_unique_states"), 'wb') as f:
            pickle.dump(self.num_unique_states, f)

def init(params):
    return SARSA(params)

def get_params():
    return ["optimizer_type", "rvfs_update_all", "action_type", "num_rvecs", "alpha", "p"]

class LinearNet(torch.nn.Module):
    def __init__(self, inputSize, outputSize, bias=False):
        super(LinearNet, self).__init__()
        self.net = torch.nn.Linear(inputSize, outputSize, bias=bias)
    
    def forward(self, x):
        return self.net(x)
