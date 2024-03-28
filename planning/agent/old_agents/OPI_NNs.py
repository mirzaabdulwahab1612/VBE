from audioop import bias
import imp
from operator import index, ne
from os import stat
import numpy as np
import torch
import torch.optim as optim

# from utils.nn_model import LinearNet
from .utils.agent_utils import *
from utils.replay_buffer import SimpleReplayBuffer
from utils.PlotDeepSea import PlotDeepSea
from utils.linear_nn import linearNet
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
        self.lambdaa = params.agent_params.lambdaa
        self.alpha = params.agent_params.alpha
        self.print_mode = params.agent_params.print_mode
        self.print_frequency = params.agent_params.print_frequency
        self.feature_constructor = params.feature_constructor
        self.np_random = params.np_random
        self.logger = params.logger

        # update frequencies
        self.rep_update_freq = params.agent_params.rep_update_freq
        self.update_freq_policy = params.agent_params.update_freq_policy
        self.num_updates = params.agent_params.num_updates
        self.mini_batch_size = params.agent_params.mini_batch_size
        self.mem_size = self.feature_dim*self.num_action

        if self.feature_constructor.real_mode:
            self.features = np.zeros(self.feature_dim)
        else:
            self.features = np.zeros(self.feature_dim)

        self.num_rvecs = params.agent_params.num_rvecs
        num_hidden_units=self.feature_dim
        self.linear_rep = params.agent_params.linear_rep

        self.opi_network = OPINet(self.feature_dim, num_hidden_units, self.num_action, self.num_rvecs, self.linear_rep)
        self.target_opi_network = copy.deepcopy(self.opi_network)

        # for e vector in the bound
        if self.linear_rep:
            self.evec = np.zeros((self.num_rvecs, self.mem_size))
        else:
            self.evec = np.zeros((self.num_rvecs, num_hidden_units*self.num_action))
        self.policy_evec = copy.deepcopy(self.evec)
        
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
        self.plotter = PlotDeepSea(params)

        # Optimizer
        for name, param in self.opi_network.named_parameters():
            print(name, param.shape)

        self.optimizer = optim.SGD(self.opi_network.parameters(), lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def start(self, observation):
        # Getting state features of next S`
        get_state(observation, self.feature_constructor, self.features)
        next_act = self.optimistic_action(self.features, self.opi_network, self.evec, greedy=False, target_action=False)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, observation, reward, terminal):

        # Getting state features of next S`
        get_state(observation, self.feature_constructor, self.features)
        # Storing data in the replay_buffer
        if(self.np_random.uniform(0,1) < 0.5):
            self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))
        else:
            self.replay_buffer_test.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))

        next_act = self.optimistic_action(self.features, self.opi_network, self.evec, greedy=False, target_action=False, print_flag=True)
        if next_act is None:
            return next_act

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.target_opi_network = copy.deepcopy(self.opi_network)
            self.policy_evec = copy.deepcopy(self.evec)

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size and self.replay_buffer_test.get_buffer_size() > self.mini_batch_size):
            self.update_representation(opi=self.target_opi_network, evec=self.policy_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_one_update(opi=self.target_opi_network, evec=self.policy_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_uncertainty_estimates(opi=self.target_opi_network, evec=self.policy_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = next_act
        self.time_step += 1

        return next_act

    
    def learn_one_update(self, opi=None, evec=None, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, opi, evec, greedy=self.target_greedy_wrt_value, target_action=True)

            terminals_ = torch.from_numpy(np.invert(terminals)).float()
            rewards = torch.from_numpy(rewards).float()
            states_t = torch.from_numpy(states).float()
            next_states_t = torch.from_numpy(next_states).float()

            hidden_state = self.opi_network.hidden_state(states_t).detach()
            hidden_state_next = self.opi_network.hidden_state(next_states_t).detach()
            next_values = self.opi_network.value(hidden_state_next).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.opi_network.value(hidden_state)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            # for name, param in self.opi_network.named_parameters():
            #     print(name, param.grad)

            # updating weights backpropagation        
            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()

    
    def update_representation(self, opi=None, evec=None, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, opi, evec, greedy=self.target_greedy_wrt_value, target_action=True)

            terminals_ = torch.from_numpy(np.invert(terminals)).float()
            rewards = torch.from_numpy(rewards).float()
            states_t = torch.from_numpy(states).float()
            next_states_t = torch.from_numpy(next_states).float()

            hidden_state = self.opi_network.hidden_state(states_t)
            
            hidden_state_next = self.opi_network.hidden_state(next_states_t).detach()
            next_values = self.opi_network.value(hidden_state_next).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.opi_network.value(hidden_state)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            self.opi_network.wvec.weight.grad = None
            # if(self.opi_network.wvec.weight.grad is not None):
            #     print(self.opi_network.wvec.weight.grad.shape)

            # for name, param in self.opi_network.named_parameters():
            #     if param.grad is not None:
            #         print(name, np.unique(param.grad.data.numpy()))
            #     else:
            #         print(name, param.grad)

            # updating weights backpropagation        
            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()

    def learn_uncertainty_estimates(self, opi=None, evec=None, num_updates=10, mini_batch_size=64):

        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, opi, evec, greedy=self.target_greedy_wrt_value, target_action=True)
            terminals_ = np.invert(terminals)

            terminals_ = torch.from_numpy(terminals_).float()
            rewards = torch.from_numpy(rewards).float()
            next_states_t = torch.from_numpy(next_states).float()
            states_t = torch.from_numpy(states).float()

            hidden_state = self.opi_network.hidden_state(states_t).detach()
            hidden_state_next = self.opi_network.hidden_state(next_states_t).detach()
            
            next_values = self.opi_network.value(hidden_state_next).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.opi_network.value(hidden_state).detach()
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            td_error = target_q - current_q

            next_values = self.opi_network.uncertainty(hidden_state_next).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.opi_network.uncertainty(hidden_state)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = td_error + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            # -- Slower version --
            for j in range(self.num_rvecs):
                next_values = self.opi_network.random_target(hidden_state_next, j).detach()
                next_q = next_values[np.arange(mini_batch_size), next_actions]
                current_values = self.opi_network.random_target(hidden_state, j).detach()
                current_q = current_values[np.arange(mini_batch_size), actions]
                reward_u = current_q - (self.gamma*terminals_*next_q)

                next_values = self.opi_network.random_prediction(hidden_state_next, j).detach()
                next_q = next_values[np.arange(mini_batch_size), next_actions]
                current_values = self.opi_network.random_prediction(hidden_state, j)
                current_q = current_values[np.arange(mini_batch_size), actions]
                target_q = reward_u + self.gamma*terminals_*next_q
                loss += self.loss_fn(current_q, target_q)

            #-- Faster version --
            # next_values = get_values(next_states.T, (self.rand_wvec - self.rvec), self.num_action, multi=True)
            # next_q = next_values[np.arange(mini_batch_size), :, next_actions]
            # current_values = get_values(states.T, (self.rand_wvec - self.rvec), self.num_action, multi=True)
            # current_q = current_values[np.arange(mini_batch_size), actions]
            # terminals_ = np.tile(terminals_, (self.num_rvecs, 1)).T    
            # td_error_w_rand = current_q - self.gamma*terminals_*next_q
            # batch_errors_w_rand = td_error_w_rand.T.dot(states)
            # self.rvec = self.rvec + ((self.alpha/mini_batch_size)* batch_errors_w_rand)
            # self.evec = self.rvec - self.rand_wvec 


            self.optimizer.zero_grad()
            loss.backward()    
            self.optimizer.step()

            self.evec = np.array([self.opi_network.rvecs[k].weight.data.flatten().numpy() - self.opi_network.rand_wvecs[k].weight.data.flatten().numpy() for k in range(self.num_rvecs)])

    
    def optimistic_action(self, state, opi, evec=None, greedy=False, target_action=False, print_flag=False):
        with torch.no_grad():
            if opi is None:
                opi = self.opi_network
            if evec is None:
                evec = self.evec

            hidden_state = torch.from_numpy(state).float()
            hidden_state = opi.hidden_state(hidden_state).detach()

            if(greedy):
                val = opi.value(hidden_state)
                value_new = val
            else:                
                val = opi.value(hidden_state).detach().numpy()
                unc = opi.uncertainty(hidden_state).detach().numpy()
                iunc = get_values(hidden_state.T, evec, self.num_action, multi=True)
                print(f"iunc: {iunc.shape}")
                if(target_action):
                    # iunc = np.max(iunc, axis=1)
                    id = np.argmax(np.abs(iunc), axis=1)
                    iunc_0 = iunc[np.arange(self.mini_batch_size), id[:, 0]][:, 0]
                    iunc_1 = iunc[np.arange(self.mini_batch_size), id[:, 1]][:, 1]
                    iunc = np.stack((iunc_0, iunc_1), axis=1)
                else:
                    # iunc = np.max(iunc, axis=0)
                    id = np.argmax(np.abs(iunc), axis=0)
                    iunc_0 = iunc[id[0]][0]
                    iunc_1 = iunc[id[1]][1]
                    iunc = np.stack((iunc_0, iunc_1), axis=0)
                
                uncertainty_bonus = self.c * np.power(np.power((unc + iunc), 2), 0.5)
                # uncertainty_bonus = self.c * np.power(np.power((iunc), 2), 0.5)
                value_new = val + uncertainty_bonus

            act = get_greedy_action(value_new, self.np_random, target_action)
            # print(f"value_new: {iunc}, act: {act}") if print_flag else None

        return act

    def save_policy(self, path):
        np.save(path+"_w",self.wvec)


def init(params):
    return SARSA(params)

def get_params():
    # return ["alpha"]
    return ["alpha", "p"]


class OPINet(torch.nn.Module):
    def __init__(self, inputSize, num_hidden_units, outputSize, num_rvecs, linear_rep=False):
        super(OPINet, self).__init__()

        self.num_rvecs = num_rvecs
        self.linear_rep = linear_rep

        # For linear case without representation network
        if self.linear_rep:
            self.wvec = torch.nn.Linear(inputSize, outputSize, bias=False)
            self.uvec = torch.nn.Linear(inputSize, outputSize, bias=False)

            # Random prior networks
            self.rand_wvecs = torch.nn.ModuleList([torch.nn.Linear(inputSize, outputSize, bias=False) for i in range(self.num_rvecs)])
            self.rvecs = torch.nn.ModuleList([torch.nn.Linear(inputSize, outputSize, bias=False) for i in range(self.num_rvecs)])

        else:
            print(f"Non-linear Representation Network")
            self.rep_net=torch.nn.Sequential(torch.nn.Linear(inputSize, num_hidden_units, bias=False))
            # self.rep_net=torch.nn.Sequential(torch.nn.Linear(inputSize, num_hidden_units),
            #                                 torch.nn.ReLU(),
            #                                 torch.nn.Linear(num_hidden_units, num_hidden_units)
            #                                 )
            self.wvec = torch.nn.Linear(num_hidden_units, outputSize, bias=False)
            self.uvec = torch.nn.Linear(num_hidden_units, outputSize, bias=False)

            # Random prior networks
            self.rand_wvecs = torch.nn.ModuleList([torch.nn.Linear(num_hidden_units, outputSize, bias=False) for i in range(self.num_rvecs)])
            self.rvecs = torch.nn.ModuleList([torch.nn.Linear(num_hidden_units, outputSize, bias=False) for i in range(self.num_rvecs)])

        
        for i, l in enumerate(self.rand_wvecs):
            torch.nn.init.normal_(l.weight)

        for i, l in enumerate(self.rvecs):
            torch.nn.init.zeros_(l.weight)

    def hidden_state(self, x):
        if self.linear_rep:
            return x
        else:
            return self.rep_net(x)

    def value(self, x):
        out = self.wvec(x)
        return out
    
    def uncertainty(self, x):
        out = self.uvec(x)
        return out

    def random_target(self, x, index):
        with torch.no_grad():
            return self.rand_wvecs[index](x)

    def random_prediction(self, x, index):
        return self.rvecs[index](x)

