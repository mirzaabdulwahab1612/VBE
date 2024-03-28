from audioop import bias
import imp
from operator import index, ne
from os import stat
import numpy as np
import torch
import torch.optim as optim

from .utils.agent_utils import *
from utils.replay_buffer import SimpleReplayBuffer
from utils.PlotDeepSea import PlotDeepSea
import copy


class SARSA():

    def __init__(self, params):

        # Replay buffer
        self.replay_buffer_train = SimpleReplayBuffer()
        self.replay_buffer_test = SimpleReplayBuffer()

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
        self.start_training = False

        # update frequencies
        self.update_freq_policy = params.agent_params.update_freq_policy
        self.num_updates = params.agent_params.num_updates
        self.mini_batch_size = params.agent_params.mini_batch_size
        self.mem_size = self.feature_dim*self.num_action

        if self.feature_constructor.real_mode:
            self.features = np.zeros(self.feature_dim)
        else:
            self.features = np.zeros(self.feature_dim)

        self.num_rvecs = params.agent_params.num_rvecs
        num_hidden_units=50

        # self.wvec = OPINet(self.feature_dim, num_hidden_units, self.num_action)
        # self.uvec = OPINet(self.feature_dim, num_hidden_units, self.num_action)

        self.wvec = LinearNet(self.feature_dim, self.num_action, zero_init=True)
        self.uvec = LinearNet(self.feature_dim, self.num_action, zero_init=True)

        self.rand_target = []
        self.rand_pred = []
        self.evec = []
        for i in range(self.num_rvecs):
            
            self.rand_target.append(LinearNet(self.feature_dim, self.num_action))
            self.rand_pred.append(LinearNet(self.feature_dim, self.num_action, zero_init=True))
            self.evec.append(LinearNet(self.feature_dim, self.num_action))
            # self.rand_target.append(OPINet(self.feature_dim, num_hidden_units, self.num_action))
            # self.rand_pred.append(OPINet(self.feature_dim, num_hidden_units, self.num_action))

        self.target_wvec = copy.deepcopy(self.wvec)
        self.target_uvec = copy.deepcopy(self.uvec)
        self.target_rand_pred = copy.deepcopy(self.rand_pred)
        self.target_evec = copy.deepcopy(self.evec)
        
        # parameters for the bound
        self.p = params.agent_params.p
        self.c = np.power(((1+self.p)/self.p), 0.5)

        # For storing current state-action-features
        self.current_state = np.zeros(self.mem_size)
        self.current_action = None
        self.time_step = 0
        self.episode_num = 0

        # True: target policy greedy wrt values. False: target policy greedy wrt values + bonus.
        self.target_greedy_wrt_value = params.agent_params.target_off_policy

        # Deep sea plotter
        # self.plotter = PlotDeepSea(params)

        net_params = list(self.wvec.parameters()) + list(self.uvec.parameters()) + list()
        for i in range(self.num_rvecs):
            net_params += list(self.rand_pred[i].parameters()) + list(self.evec[i].parameters())
        # Optimizer
        self.optimizer = optim.SGD(net_params, lr=self.alpha)
        # self.optimizer = optim.Adam(net_params, lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def start(self, observation):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.rand_pred, self.evec, greedy=False, target_action=False)

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

        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.rand_pred, self.evec, greedy=False, target_action=False, print_flag=True)
        if next_act is None:
            return next_act

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.target_wvec = copy.deepcopy(self.wvec)
            self.target_uvec = copy.deepcopy(self.uvec)
            self.target_rand_pred = copy.deepcopy(self.rand_pred)
            self.target_evec = copy.deepcopy(self.evec)

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size and self.replay_buffer_test.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            self.learn_one_update(wvec=self.target_wvec, uvec=self.target_uvec, rand_pred=self.target_rand_pred, evec=self.target_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_uncertainty_estimates(wvec=self.target_wvec, uvec=self.target_uvec, rand_pred=self.target_rand_pred, evec=self.target_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = next_act
        self.time_step += 1

        # for k in range (self.feature_dim):
        #     get_state([k], self.feature_constructor, self.features)
        #     temp_state = torch.from_numpy(copy.deepcopy(self.features)).float()
        #     temp_v = self.wvec.forward(temp_state).detach().numpy()
        #     temp_u = self.uvec.forward(temp_state).detach().numpy()
        #     temp_iu = self.evec[0].forward(temp_state).detach().numpy()
        #     print("Observation: ", k, " Value: ", temp_v, " Uncertainty: ", temp_u, " IU: ", temp_iu)

        # exit(1)

        return next_act

    
    def learn_one_update(self, wvec=None, uvec=None, rand_pred=None, evec=None, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, wvec, uvec, rand_pred, evec, greedy=self.target_greedy_wrt_value, target_action=True)

            terminals_ = torch.from_numpy(np.invert(terminals)).float()
            rewards = torch.from_numpy(rewards).float()
            states_t = torch.from_numpy(states).float()
            next_states_t = torch.from_numpy(next_states).float()

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

    def learn_uncertainty_estimates(self, wvec=None, uvec=None, rand_pred=None, evec=None, num_updates=10, mini_batch_size=64):

        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, wvec, uvec, rand_pred, evec, greedy=self.target_greedy_wrt_value, target_action=True)
            terminals_ = np.invert(terminals)

            terminals_ = torch.from_numpy(terminals_).float()
            rewards = torch.from_numpy(rewards).float()
            next_states_t = torch.from_numpy(next_states).float()
            states_t = torch.from_numpy(states).float()
            
            next_values = self.wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t).detach()
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            td_error = target_q - current_q

            next_values = self.uvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.uvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = td_error + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            # -- Slower version --
            for j in range(self.num_rvecs):
                next_values = self.rand_target[j].forward(next_states_t).detach()
                next_q = next_values[np.arange(mini_batch_size), next_actions]
                current_values = self.rand_target[j].forward(states_t).detach()
                current_q = current_values[np.arange(mini_batch_size), actions]
                reward_u = current_q - (self.gamma*terminals_*next_q)

                next_values = self.rand_pred[j].forward(next_states_t).detach()
                next_q = next_values[np.arange(mini_batch_size), next_actions]
                current_values = self.rand_pred[j].forward(states_t)
                current_q = current_values[np.arange(mini_batch_size), actions]
                target_q = reward_u + self.gamma*terminals_*next_q
                loss += self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()

            loss = 0
            for j in range(self.num_rvecs):
                next_values = self.rand_target[j].forward(next_states_t).detach()
                next_target = next_values[np.arange(mini_batch_size), next_actions]
                next_values = self.rand_pred[j].forward(next_states_t).detach()
                next_pred = next_values[np.arange(mini_batch_size), next_actions]

                current_values = self.rand_target[j].forward(states_t).detach()
                current_target = current_values[np.arange(mini_batch_size), actions]
                current_values = self.rand_pred[j].forward(states_t).detach()
                current_pred = current_values[np.arange(mini_batch_size), actions]

                # next_state_error = torch.abs(next_target - next_pred)
                next_state_error = torch.abs(next_pred - next_target)
                # current_state_error = torch.abs(current_target - current_pred)
                current_state_error = torch.abs(current_pred - current_target)
                reward_error = current_state_error - (self.gamma*terminals_*next_state_error)
                
                next_values = self.evec[j].forward(next_states_t).detach()
                next_q = next_values[np.arange(mini_batch_size), next_actions]
                current_values = self.evec[j].forward(states_t)
                current_q = current_values[np.arange(mini_batch_size), actions]
                target_q = reward_error + self.gamma*terminals_*next_q
                loss += self.loss_fn(current_q, target_q)
                
            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()


    
    def optimistic_action(self, state, wvec=None, uvec=None, rand_pred=None, evec=None, greedy=False, target_action=False, print_flag=False):
        with torch.no_grad():
            if wvec is None:
                wvec = self.wvec
            if uvec is None:
                uvec = self.uvec
            if rand_pred is None:
                rand_pred = self.rand_pred
            if evec is None:
                evec = self.evec

            hidden_state = torch.from_numpy(state).float()

            if self.start_training:
                if(greedy):
                    val = wvec.forward(hidden_state)
                    value_new = val
                else:                
                    val = wvec.forward(hidden_state).detach().numpy()
                    unc = uvec.forward(hidden_state).detach().numpy()
                    iunc = np.array([evec[id].forward(hidden_state).detach().numpy() for id in range(self.num_rvecs)])
                    # print("iunc", iunc.shape)

                    # rand_p = np.array([rand_pred[id].forward(hidden_state).detach().numpy() for id in range(self.num_rvecs)])
                    # rand_t = np.array([self.rand_target[id].forward(hidden_state).detach().numpy() for id in range(self.num_rvecs)])
                    # iunc = rand_p - rand_t
                    # if(len(iunc.shape) == 3):
                    #     iunc = np.swapaxes(iunc,0,1)

                    temp_iunc = []
                    if(target_action):
                        iunc = np.swapaxes(iunc,0,1)
                        id = np.argmax(np.abs(iunc), axis=1)
                        for k in range(self.num_action):
                            temp_iunc.append(iunc[np.arange(self.mini_batch_size), id[:, k]][:, k])
                        iunc = np.stack(temp_iunc, axis=1)
                    else:
                        # iunc = np.max(iunc, axis=0)
                        id = np.argmax(np.abs(iunc), axis=0)
                        for k in range(self.num_action):
                            temp_iunc.append(iunc[id[k]][k])
                        iunc = np.stack(temp_iunc, axis=0)
                    
                    uncertainty_bonus = self.c * np.power(np.power((unc + iunc), 2), 0.5)
                    # uncertainty_bonus = self.c * np.power(np.power((iunc), 2), 0.5)
                    value_new = val + uncertainty_bonus

            else:
                value_new = np.zeros((self.num_action))

            
            act = get_greedy_action(value_new, self.np_random, target_action)

        return act

    def save_policy(self, path):
        np.save(path+"_w",self.wvec)


def init(params):
    return SARSA(params)

def get_params():
    # return ["alpha"]
    return ["alpha", "p"]


class OPINet(torch.nn.Module):
    def __init__(self, inputSize, num_hidden_units, outputSize, zero_init=False):
        super(OPINet, self).__init__()

        self.net=torch.nn.Sequential(torch.nn.Linear(inputSize, num_hidden_units),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(num_hidden_units, num_hidden_units),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(num_hidden_units, outputSize)
                                    )
        if(zero_init):
            for param in self.net.parameters():
                param.data.fill_(0)
    
    def forward(self, x):
        return self.net(x)

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
