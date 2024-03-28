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

class SARSA():

    def __init__(self, params):

        # Replay buffer
        self.replay_buffer_train = SimpleReplayBuffer(params.np_random)

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
        self.wvec = LinearNet(self.feature_dim, self.num_action, bias=True)
        self.target_wvec = LinearNet(self.feature_dim, self.num_action, bias=True)
        self.intrinsic_wvec = LinearNet(self.feature_dim, self.num_action, bias=True)
        self.target_intrinsic_wvec = LinearNet(self.feature_dim, self.num_action, bias=True)

        # outofsample linear heads
        self.rnd_target = LinearNet(self.feature_dim, self.num_rvecs, bias=True)
        self.rnd_pred = LinearNet(self.feature_dim, self.num_rvecs, bias=True)

        # Target networks
        self.target_wvec.load_state_dict(self.wvec.state_dict())
        self.target_intrinsic_wvec.load_state_dict(self.intrinsic_wvec.state_dict())

        # parameters for the bound
        self.p = params.agent_params.p
        self.target_polic_type = params.agent_params.target_polic_type

        # For storing current state-action-features
        self.current_state = np.zeros(self.mem_size)
        self.current_action = None
        self.time_step = 0
        self.episode_num = 0
        self.start_training = False

        net_params = list(self.wvec.parameters()) + list(self.intrinsic_wvec.parameters()) + list(self.rnd_pred.parameters())
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

    def start(self, observation):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.intrinsic_wvec, target_action=False)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, observation, reward, terminal):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        # Storing data in the replay_buffer
        self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))
        next_act = self.optimistic_action(self.features, self.wvec, self.intrinsic_wvec, target_action=False)

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.target_wvec.load_state_dict(self.wvec.state_dict())
            self.target_intrinsic_wvec.load_state_dict(self.intrinsic_wvec.state_dict())

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1

        return next_act
    
    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            if self.target_polic_type == "optimistic" or self.target_polic_type == "optimistic-uniform":
                next_actions = self.optimistic_action(next_states, self.wvec, self.intrinsic_wvec, target_action=True)
            elif self.target_polic_type == "greedy" or self.target_polic_type == "greedy-greedy" or self.target_polic_type == "greedy-uniform":
                next_actions = np.argmax(self.wvec.forward(next_states_t).detach(), axis=-1)
            else:
                exit("Invalid value target policy type")
            next_values = self.target_wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            extrinsic_loss = self.loss_fn(current_q, target_q)

            # updating intrinsic value function
            target_next_feature = self.rnd_target(next_states_t).detach()
            predict_next_feature = self.rnd_pred(next_states_t).detach()
            intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
            next_values = self.target_intrinsic_wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.intrinsic_wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = intrinsic_reward + self.gamma*next_q
            intrinsic_loss = self.loss_fn(current_q, target_q)

            # updating the auxiliary weights
            target_feature = self.rnd_target(states_t).detach()
            predict_feature = self.rnd_pred(states_t)
            auxLoss = self.loss_fn(predict_feature, target_feature)

            # adding all loss terms
            loss = extrinsic_loss + intrinsic_loss + auxLoss

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
    
    def optimistic_action(self, state, wvec=None, intrinsic_wvec=None, target_action=False):
        with torch.no_grad():
            hidden_state = torch.Tensor(state).to(self.device)
            if self.start_training:
                val = wvec.forward(hidden_state)
                intrinsic_val = intrinsic_wvec.forward(hidden_state)
                value_new = (val + self.p*intrinsic_val).data.cpu().numpy()
            else:
                value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                
            act = get_greedy_action(value_new, self.np_random, target_action)

        return act

    def save_policy(self, path):
        torch.save(self.wvec.state_dict(), str(path+"weights_at_step_"+str(self.time_step)))

def init(params):
    return SARSA(params)

def get_params():
    return ["optimizer_type", "target_polic_type", "num_rvecs", "alpha", "p"]

class LinearNet(torch.nn.Module):
    def __init__(self, inputSize, outputSize, bias=True):
        super(LinearNet, self).__init__()
        self.net = torch.nn.Linear(inputSize, outputSize, bias=bias)
    
    def forward(self, x):
        return self.net(x)
