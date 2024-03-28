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
        self.replay_buffer_test = SimpleReplayBuffer(params.np_random)

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
        self.wvec = LinearNet(self.feature_dim, self.num_action, zero_init=False)
        self.target_wvec = LinearNet(self.feature_dim, self.num_action, zero_init=False)

        self.prior_mean = params.agent_params.prior_mean
        if self.prior_mean > 0.0:
            self.prior_scale = 1.0
        else:
            self.prior_scale = params.agent_params.prior_scale

        self.rand_target = []
        self.rand_pred = []
        self.target_rand_pred = []
        for i in range(self.num_rvecs):
            # Linear function approximator
            self.rand_target.append(LinearNet(self.feature_dim, self.num_action, zero_init=False))
            self.rand_pred.append(PriorNet(self.feature_dim, self.num_action, prior_mean=self.prior_mean, prior_scale=self.prior_scale))
            self.target_rand_pred.append(PriorNet(self.feature_dim, self.num_action, prior_mean=self.prior_mean, prior_scale=self.prior_scale))

        # Target networks
        self.target_wvec.load_state_dict(self.wvec.state_dict())
        for i in range(self.num_rvecs):
            self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())
        
        # parameters for the bound
        self.p = params.agent_params.p
        self.c = np.power(((1+self.p)/self.p), 0.5)

        # For storing current state-action-features
        self.current_state = np.zeros(self.mem_size)
        self.current_action = None
        self.time_step = 0
        self.episode_num = 0
        self.start_training = False

        net_params = list(self.wvec.parameters())
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

        # self.vis_data = set()
        # self.vis_features = np.zeros(self.feature_dim)

    def start(self, observation):
        # self.current_position = float(observation[:])
        # self.vis_data.add(float(observation[:]))
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.rand_pred, target_action=False)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, observation, reward, terminal):
        # self.current_position = float(observation[:])
        # self.vis_data.add(float(observation[:]))
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        # Storing data in the replay_buffer
        if(self.np_random.uniform(0,1) < 0.5):
            self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))
        else:
            self.replay_buffer_test.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))

        next_act = self.optimistic_action(self.features, self.wvec, self.rand_pred, target_action=False, print_flag=True)

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.target_wvec.load_state_dict(self.wvec.state_dict())
            for i in range(self.num_rvecs):
                self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())

        # Visualising value functions
        # self.visualize_frequency = 1000
        # if self.time_step > 1200000 and self.time_step%self.visualize_frequency == 0:
        #     self.visualize()

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size and self.replay_buffer_test.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            self.learn_out_uncertainty_estimates(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1

        return next_act

    def visualize(self):
        with torch.no_grad():
            ax = plt.subplot()
            # ax.set_ylim([0, 30])
            ax.set_xlim([-0.2, 1.2])
            x_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            ax.set_xticks(x_ticks)
            vis_features = {}
            for obs in self.vis_data:
                get_features(np.array([obs]), self.feature_constructor, self.vis_features, self.vis_features)
                vis_features[tuple(copy.deepcopy(self.vis_features))] = obs

            ax.plot(self.current_position, 1, 'o', color='black')
            for keys, data in vis_features.items():
                observation = torch.Tensor(keys).to(self.device)
                rand_p = np.array([self.rand_pred[id].forward(observation).cpu().numpy() for id in range(self.num_rvecs)])
                rand_t = np.array([self.rand_target[id].forward(observation).cpu().numpy() for id in range(self.num_rvecs)])
                iunc = np.abs(rand_p - rand_t)[0]

                val = self.wvec(observation).cpu().numpy()
                uncertainty_bonus = self.c * iunc
                value_new = (val + uncertainty_bonus)
                # print(f"val: {val}, iunc: {iunc}, iunc_bonus: {iunc_bonus}, uncertainty_bonus: {uncertainty_bonus} value_new: {value_new}")

                # ax.bar(data-0.01, val[0], width=0.01, color='b', align='center')
                # ax.bar(data+0.01, val[1], width=0.01, color='g', align='center')
                # ax.bar(data-0.01, value_new[0], width=0.01, color='b', align='center')
                # ax.bar(data+0.01, value_new[1], width=0.01, color='g', align='center')
                ax.bar(data-0.01, uncertainty_bonus[0], width=0.01, color='r', align='center')
                ax.bar(data+0.01, uncertainty_bonus[1], width=0.01, color='y', align='center')
            plt.pause(0.001)
            plt.close()
    
    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.target_wvec, self.target_rand_pred, target_action=True)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            next_values = self.wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            # updating weights backpropagation        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn_out_uncertainty_estimates(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.target_wvec, self.target_rand_pred, target_action=True)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)
        
            loss = 0
            # -- Slower version --
            for j in range(self.num_rvecs):
                next_values_target = self.rand_target[j].forward(next_states_t).detach()
                next_q_target = next_values_target[np.arange(mini_batch_size), next_actions]
                current_values_target = self.rand_target[j].forward(states_t).detach()
                current_q_target = current_values_target[np.arange(mini_batch_size), actions]
                reward_u = current_q_target - (self.gamma*terminals_*next_q_target)

                next_values_pred = self.rand_pred[j].forward(next_states_t).detach()
                next_q_pred = next_values_pred[np.arange(mini_batch_size), next_actions]
                current_values_pred = self.rand_pred[j].forward(states_t)
                current_q_pred = current_values_pred[np.arange(mini_batch_size), actions]
                target_q_pred = reward_u + self.gamma*terminals_*next_q_pred
                loss += self.loss_fn(current_q_pred, target_q_pred)

            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()
    
    def optimistic_action(self, state, wvec=None,  rand_pred=None, target_action=False, print_flag=False):
        with torch.no_grad():
            hidden_state = torch.Tensor(state).to(self.device)
            if self.start_training:
                val = wvec.forward(hidden_state)

                rand_p = torch.stack([rand_pred[id].forward(hidden_state) for id in range(self.num_rvecs)])
                rand_t = torch.stack([self.rand_target[id].forward(hidden_state) for id in range(self.num_rvecs)])
                iunc = torch.abs(rand_p - rand_t)

                temp_iunc = []
                if(target_action):
                    iunc = torch.swapaxes(iunc,0,1)
                    id = torch.argmax(iunc, axis=1)
                    for k in range(self.num_action):
                        temp_iunc.append(iunc[torch.arange(len(iunc)), id[:, k]][:, k])
                    iunc = torch.stack(temp_iunc, axis=1)
                else:
                    id = torch.argmax(iunc, axis=0)
                    for k in range(self.num_action):
                        temp_iunc.append(iunc[id[k]][k])
                    iunc = torch.stack(temp_iunc, axis=0)

                # iunc_bonus = torch.nn.functional.softmax(iunc, dim=-1)
                uncertainty_bonus = self.c * (iunc)
                value_new = (val + uncertainty_bonus).data.cpu().numpy()
            else:
                value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                
            act = get_greedy_action(value_new, self.np_random, target_action)

        return act

    def save_policy(self, path):
        np.save(path+"_w",self.wvec)

def init(params):
    return SARSA(params)

def get_params():
    return ["optimizer_type", "num_rvecs", "alpha", "prior_mean", "prior_scale", "p"]

class LinearNet(torch.nn.Module):
    def __init__(self, inputSize, outputSize, prior_mean=0.0, prior_variance=1.0, zero_init=False):
        super(LinearNet, self).__init__()

        self.net = torch.nn.Linear(inputSize, outputSize, bias=True)
                                    
        if(zero_init):
            for param in self.net.parameters():
                param.data.fill_(0)
        else:
            torch.nn.init.normal_(self.net.weight, mean=prior_mean, std=prior_variance)
            # torch.nn.init.normal_(self.net.weight)
            # torch.nn.init.trunc_normal_(self.net.weight, mean=prior_mean, std=(1.0/np.power(inputSize, 1/2)), a=prior_mean-2, b=prior_mean+2)
    
    def forward(self, x):
        return self.net(x)

class PriorNet(torch.nn.Module):
    def __init__(self, inputSize, outputSize, prior_mean=0.0, prior_scale=3.0, zero_init=False):
        super(PriorNet, self).__init__()

        self._prior_scale = prior_scale
        self.net = torch.nn.Linear(inputSize, outputSize, bias=True)
        self.prior_net = torch.nn.Linear(inputSize, outputSize, bias=True)
                                    
        if(zero_init):
            for param in self.net.parameters():
                param.data.fill_(0)
        else:
            torch.nn.init.trunc_normal_(self.net.weight, mean=0.0, std=(1.0/np.power(inputSize, 1/2)), a=-2, b=2)
        
        # prior_net is initialized to have optimism
        torch.nn.init.trunc_normal_(self.prior_net.weight, mean=prior_mean, std=(1.0/np.power(inputSize, 1/2)), a=prior_mean-2, b=prior_mean+2)
    
    def forward(self, x):
        q_values = self.net(x)
        prior_q_values = self.prior_net(x).detach()
        return q_values + (self._prior_scale * prior_q_values)