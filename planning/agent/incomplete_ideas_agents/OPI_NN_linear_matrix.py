from audioop import bias
import imp
from operator import index, ne
from os import stat
import numpy as np
import torch
import torch.optim as optim
from scipy.special import softmax

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
        self.wvec = LinearNet(self.feature_dim, self.num_action, zero_init=True)
        self.uvec = LinearNet(self.feature_dim, self.num_action, zero_init=True)
        self.target_wvec = LinearNet(self.feature_dim, self.num_action, zero_init=True)
        self.target_uvec = LinearNet(self.feature_dim, self.num_action, zero_init=True)

        self.rand_target = self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size), size=self.num_rvecs)
        print(f"self.rand_target: {self.rand_target}")
        # self.rand_target = []
        self.rand_pred = []
        self.target_rand_pred = []
        for i in range(self.num_rvecs):
            # Linear function approximator
            # self.rand_target.append(LinearNet(self.feature_dim, self.num_action))
            self.rand_pred.append(LinearNet(self.feature_dim, self.num_action, zero_init=True))
            self.target_rand_pred.append(LinearNet(self.feature_dim, self.num_action, zero_init=True))

        # Target networks
        self.target_wvec.load_state_dict(self.wvec.state_dict())
        self.target_uvec.load_state_dict(self.uvec.state_dict())
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

        net_params = list(self.wvec.parameters()) + list(self.uvec.parameters())
        for i in range(self.num_rvecs):
            net_params += list(self.rand_pred[i].parameters())
        # Optimizer
        self.optimizer = optim.SGD(net_params, lr=self.alpha, weight_decay=0.0)
        # self.optimizer = optim.Adam(net_params, lr=self.alpha, weight_decay=0.0)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def start(self, observation):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.rand_pred, target_action=False)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, observation, reward, terminal):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        # Storing data in the replay_buffer
        if(self.np_random.uniform(0,1) < 0.5):
            self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))
        else:
            self.replay_buffer_test.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))

        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.rand_pred, target_action=False, print_flag=True)
        if next_act is None:
            return next_act

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.target_wvec.load_state_dict(self.wvec.state_dict())
            self.target_uvec.load_state_dict(self.uvec.state_dict())
            for i in range(self.num_rvecs):
                self.target_rand_pred[i].load_state_dict(self.rand_pred[i].state_dict())

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size and self.replay_buffer_test.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            self.learn_out_uncertainty_estimates(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_uncertainty_estimates(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            # self.learn_out_uncertainty_estimates(num_updates=1, mini_batch_size=1)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = next_act
        self.time_step += 1

        return next_act

    
    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.target_wvec, self.target_uvec, self.target_rand_pred, target_action=True)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            next_values = self.wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)/2

            # updating weights backpropagation        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn_uncertainty_estimates(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.target_wvec, self.target_uvec, self.target_rand_pred, target_action=True)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            next_values = self.wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t).detach()
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            td_error = target_q - current_q

            next_values_u = self.uvec.forward(next_states_t).detach()
            next_q_u = next_values_u[np.arange(mini_batch_size), next_actions]
            current_values_u = self.uvec.forward(states_t)
            current_q_u = current_values_u[np.arange(mini_batch_size), actions]
            target_q_u = td_error + self.gamma*terminals_*next_q_u
            loss = self.loss_fn(current_q_u, target_q_u)/2

            self.optimizer.zero_grad()
            loss.backward()   
            self.optimizer.step()

    def learn_out_uncertainty_estimates(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.target_wvec, self.target_uvec, self.target_rand_pred, target_action=True)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)
        
            # -- Slower version --
            for j in range(self.num_rvecs):
                next_values_target = get_values(next_states, self.rand_target[j], self.num_action)
                next_q_target = next_values_target[np.arange(mini_batch_size), next_actions]
                current_values_target = get_values(states, self.rand_target[j], self.num_action)
                current_q_target = current_values_target[np.arange(mini_batch_size), actions]

                next_q_target = torch.Tensor(next_q_target).to(self.device)
                current_q_target = torch.Tensor(current_q_target).to(self.device)

                reward_u = current_q_target - (self.gamma*terminals_*next_q_target)

                next_values_pred = self.rand_pred[j].forward(next_states_t).detach()
                next_q_pred = next_values_pred[np.arange(mini_batch_size), next_actions]
                current_values_pred = self.rand_pred[j].forward(states_t)
                current_q_pred = current_values_pred[np.arange(mini_batch_size), actions]
                target_q_pred = reward_u + self.gamma*terminals_*next_q_pred
                loss = self.loss_fn(current_q_pred, target_q_pred)/2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def optimistic_action(self, state, wvec=None, uvec=None, rand_pred=None, target_action=False, print_flag=False):
        with torch.no_grad():
            hidden_state = torch.Tensor(state).to(self.device)
            if self.start_training:
                val = wvec.forward(hidden_state)
                unc = uvec.forward(hidden_state)

                rand_p = torch.stack([rand_pred[id].forward(hidden_state) for id in range(self.num_rvecs)])
                rand_t = np.array([get_values(state, self.rand_target[id], self.num_action) for id in range(self.num_rvecs)])
                rand_t = torch.Tensor(rand_t).to(self.device)
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

                # unc_bonus = torch.nn.functional.softmax(unc, dim=-1)
                # iunc_bonus = torch.nn.functional.softmax(iunc, dim=-1)
                unc_bonus = softmax(unc, axis=-1)
                iunc_bonus = softmax(iunc, axis=-1)

                uncertainty_bonus = self.c * (unc_bonus + iunc_bonus)
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
    return ["alpha", "p"]

class LinearNet(torch.nn.Module):
    def __init__(self, inputSize, outputSize, zero_init=False):
        super(LinearNet, self).__init__()

        self.net = torch.nn.Linear(inputSize, outputSize, bias=False)
                                    
        if(zero_init):
            for param in self.net.parameters():
                param.data.fill_(0)
        else:
            torch.nn.init.normal_(self.net.weight)
    
    def forward(self, x):
        return self.net(x)
