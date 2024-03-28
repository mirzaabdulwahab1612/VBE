from importlib import import_module
import numpy as np
from .utils.agent_utils import *
from utils.dummy import DummyObject
from utils.nn_model import *
import itertools
import torch
from utils.PlotDeepSea import PlotDeepSea



class BootDQN():

    def __init__(self, params):

        self.num_action = params.environment.num_action
        self.obs_dim = params.environment.obs_dim
        self.obs_limits = params.environment_params.obs_limits
        self.feature_dim = params.feature_constructor.feature_dim

        self._noise_scale = params.agent_params.noise_scale
        self.gamma = params.agent_params.gamma
        self.weight_reward = params.agent_params.weight_reward
        self.p = params.agent_params.p
        self._prior_scale = params.agent_params.prior_scale
        self.alpha = params.agent_params.alpha
        self.num_heads = params.agent_params.num_heads
        self.buffer_size = params.agent_params.buffer_size
        self.nonlinear_rep = params.agent_params.nonlinear_rep
        self.policy_update_frequency = params.agent_params.policy_update_frequency
        self.print_mode = params.agent_params.print_mode
        self.print_frequency = params.agent_params.print_frequency

        self.feature_constructor = params.feature_constructor

        self.np_random = params.np_random
        self.logger = params.logger

        self.mem_size = self.feature_dim*self.num_action

        self.values = np.zeros(self.num_action)
        self.prior_values = np.zeros(self.num_action)

        if self.feature_constructor.real_mode:
            self.feature_count = self.feature_dim
        else:
            self.feature_count = self.feature_constructor.sparse_feature_size
        self.features = np.zeros(self.feature_dim)
        self.features_vec_state = np.zeros(self.feature_dim)
        self.features_vec = np.zeros(self.mem_size)

        self.current_head = self.np_random.choice(self.num_heads,1)[0]

        if self.nonlinear_rep:
            self.device = params.device
            self.dtype = params.dtype
            self.dtype_long =  params.dtype_long

            self.wnets = []
            self.optimizer_wnets = []
            self.wnets_target = []
            self.wvecs = []
            for i in range(self.num_heads):
                self.wnets.append(LinearNet([self.feature_dim,self.num_action],params.feature_constructor.use_bias,use_uniform_init=True).to(self.device))
                self.wvecs.append(np.zeros(self.mem_size))
                self.optimizer_wnets.append(torch.optim.Adam(self.wnets[-1].parameters(), lr=params.feature_constructor_params.lr_nn, amsgrad=params.feature_constructor_params.amsgrad, betas=(params.feature_constructor_params.beta1, params.feature_constructor_params.beta2)))
                self.wnets_target.append(LinearNet([self.feature_dim,self.num_action],params.feature_constructor.use_bias,use_uniform_init=True).to(self.device))

            self.error = torch.nn.MSELoss()

        else:

            self.update_vec = np.zeros([self.mem_size])
            self.wvecs = []
            self.wvecs_prior = []
            for i in range(self.num_heads):
                self.wvecs.append(self.np_random.rand(self.mem_size))
                # self.wvecs.append(np.ones(self.mem_size))
                self.wvecs_prior.append(self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size)))
            
            self.wvecs_target = copy.deepcopy(self.wvecs)
            self.target_wvecs_prior = copy.deepcopy(self.wvecs_prior)

        self.current_observation = np.zeros(self.obs_dim)
        self.current_action = None
        self.time_step = 0

        self.current_data = DummyObject()
        self.current_data.current_observation = np.zeros((self.buffer_size,self.obs_dim))
        self.current_data.current_action = np.zeros(self.buffer_size)
        self.current_data.next_observation = np.zeros((self.buffer_size,self.obs_dim))
        self.current_data.next_reward = np.zeros(self.buffer_size)
        self.current_data.next_terminal = np.zeros(self.buffer_size, dtype=bool)
        if not self.nonlinear_rep:
            self.current_data.current_state_representation = np.zeros((self.buffer_size,self.mem_size))
            self.current_data.next_state_representation = np.zeros((self.buffer_size,self.feature_dim))
        self.current_data.flags = np.zeros((self.buffer_size,self.num_heads))
        self.current_data.reward_noise = np.zeros((self.buffer_size,self.num_heads))
        
        self.current_pos = 0
        self.buffer_full = False

        self.batch_size = 128

        # Deep sea plotter
        # self.plotter = PlotDeepSea(params, "right")

    def policy_state(self, state):

        # get_values_state(self.feature_constructor, state, self.wvecs[self.current_head], self.values)
        self.values = get_values(state, self.wvecs[self.current_head], self.num_action)
        self.prior_values = get_values(state, self.wvecs_prior[self.current_head], self.num_action)

        # for i in range(self.values.size):
        #     self.logger.info("Values:{},{},{}".format(str(self.time_step), str(i), str(self.values[i])))
        # act = get_greedy_action(self.values, self.np_random)

        # act = get_greedy_action(self.values, self.np_random)
        act = get_greedy_action((self.values + (self._prior_scale * self.prior_values)), self.np_random)
        if (self.print_mode and self.time_step%self.print_frequency==0) or (np.isnan(self.values).any()):
            for i in range(self.values.size):
                self.logger.info("Values:{},{},{}".format(str(self.time_step), str(i), str(self.values[i])))

                # plotting value functions
                # self.plotter.update_plot(copy.deepcopy(np.mean(self.wvecs, axis=0)[self.feature_dim:self.feature_dim+self.feature_dim]))
                
            self.logger.info("Action:{}".format(str(act)))

        return act


    def start(self, observation):
        
        get_state(observation, self.feature_constructor, self.features)
        self.observation = observation
        next_act = self.policy_state(self.features)
        get_features_state(self.features_vec, self.features, next_act, self.feature_constructor.feature_dim)

        self.current_data.current_observation[self.current_pos,:] = observation
        self.current_data.current_action[self.current_pos] = next_act
        if not self.nonlinear_rep:
            self.current_data.current_state_representation[self.current_pos,:] = np.copy(self.features_vec)
        self.current_data.flags[self.current_pos,:] = self.np_random.binomial(1, self.p, self.num_heads)
        
        self.current_observation = observation
        self.current_action = next_act
        self.time_step += 1

        return next_act

    def step(self, observation, reward, terminal):

        if self.weight_reward:
            reward *= (1.0-self.gamma)

        self.current_data.next_observation[self.current_pos,:] = observation
        self.current_data.next_reward[self.current_pos] = reward
        self.current_data.next_terminal[self.current_pos] = terminal
        get_state(observation, self.feature_constructor, self.features)

        if not self.nonlinear_rep:
            self.current_data.next_state_representation[self.current_pos,:] = np.copy(self.features)

        self.update_weights()

        self.current_pos += 1
        if self.current_pos == self.buffer_size:
            if not self.buffer_full:
                self.buffer_full = True
            self.current_pos = 0

        self.observation = observation
        if (self.policy_update_frequency != -1 and self.time_step%self.policy_update_frequency == 0) or terminal:
            self.current_head = self.np_random.choice(self.num_heads,1)[0]
        next_act = self.policy_state(self.features)
        get_features_state(self.features_vec, self.features, next_act, self.feature_constructor.feature_dim)

        self.current_data.current_observation[self.current_pos,:] = observation
        self.current_data.current_action[self.current_pos] = next_act
        if not self.nonlinear_rep:
            self.current_data.current_state_representation[self.current_pos,:] = np.copy(self.features_vec)
        self.current_data.flags[self.current_pos,:] = self.np_random.binomial(1, self.p, self.num_heads)

        # Reward noise
        reward_noise = self.np_random.randn(self.num_heads).astype(np.float32) * self._noise_scale
        self.current_data.reward_noise[self.current_pos,:] = reward_noise

        self.current_observation = observation
        self.current_action = next_act

        self.time_step += 1

        if self.time_step%4 == 0:
            for i in range(self.num_heads):
                self.wvecs_target[i][:] = self.wvecs[i]

        return next_act

    def update_weights_nonlinear(self):

        return

    def update_weights(self):

        if self.nonlinear_rep:
            self.update_weights_nonlinear()
            return

        batch_size = self.batch_size
        if self.buffer_full:
            data_state_all = self.current_data.current_state_representation
            data_next_state_all = self.current_data.next_state_representation
            next_reward_all = self.current_data.next_reward
            next_terminal_all = np.logical_not(self.current_data.next_terminal)
            flags = self.current_data.flags
            reward_noise_all = self.current_data.reward_noise
            size = self.buffer_size
        else:
            data_state_all = self.current_data.current_state_representation[:self.current_pos+1,:]
            data_next_state_all = self.current_data.next_state_representation[:self.current_pos+1,:]
            next_reward_all = self.current_data.next_reward[:self.current_pos+1]
            next_terminal_all = np.logical_not(self.current_data.next_terminal[:self.current_pos+1])
            flags = self.current_data.flags[:self.current_pos+1,:]
            reward_noise_all = self.current_data.reward_noise[:self.current_pos+1,:]
            size = self.current_pos+1
            if batch_size > size:
                batch_size = size

        if self.p == 1:

            for i in range(self.num_heads):
                indices = self.np_random.choice(size,batch_size,replace=False)

                data_state = data_state_all[indices,:]
                data_next_state = data_next_state_all[indices,:]
                next_reward = next_reward_all[indices]
                next_terminal = next_terminal_all[indices]
                reward_noise = reward_noise_all[indices,:]

                current_pred = data_state.dot(self.wvecs[i])

                values_next = []
                for i in range(self.num_action):
                    offset = int(self.feature_dim*i)
                    values_next.append(data_next_state.dot(self.wvecs_target[i][offset:offset+self.feature_dim])[...,np.newaxis])
                    # values_next.append(data_next_state.dot(self.wvecs[i][offset:offset+self.feature_dim])[...,np.newaxis])
                values_next = np.concatenate(values_next,axis=1)

                next_action = None
                next_action=[get_greedy_action(values_next[i,:], self.np_random) for i in range(batch_size)]
                next_action = np.array(next_action)
                next_pred = values_next[np.arange(batch_size), next_action]

                td_error = (next_reward + reward_noise) + np.multiply(next_terminal*self.gamma,next_pred) - current_pred
                self.update_vec[:] = data_state.T.dot(td_error).squeeze()
                self.wvecs[i] += ((self.alpha/batch_size)*self.update_vec)

        else:

            for i in range(self.num_heads):
                batch_size = self.batch_size

                rel_indices = np.argwhere(flags[:,i] == 1)

                size = rel_indices.shape[0]
                if size == 0:
                    continue

                if batch_size > size:
                    batch_size = size

                
                indices = self.np_random.choice(size,batch_size,replace=False)
                data_state = data_state_all[rel_indices[indices],:]
                data_next_state = data_next_state_all[rel_indices[indices],:]
                next_reward = next_reward_all[rel_indices[indices]]
                next_terminal = next_terminal_all[rel_indices[indices]]
                reward_noise = reward_noise_all[rel_indices[indices],:]

                current_pred = data_state.dot((self.wvecs[i] + (self._prior_scale * self.wvecs_prior[i])))

                values_next = []
                for j in range(self.num_action):
                    offset = int(self.feature_dim*j)
                    values_next.append(data_next_state.dot((self.wvecs_target[i][offset:offset+self.feature_dim] + (self._prior_scale * self.target_wvecs_prior[i][offset:offset+self.feature_dim])))[...,np.newaxis])
                    # values_next.append(data_next_state.dot(self.wvecs[i][offset:offset+self.feature_dim])[...,np.newaxis])
                values_next = np.concatenate(values_next,axis=1)

                # values_next = get_values(data_next_state, self.wvecs_target[i], self.num_action)
                next_action = None
                next_action=[get_greedy_action(values_next[i,:], self.np_random) for i in range(batch_size)]
                next_action = np.array(next_action)
                next_pred = values_next[np.arange(batch_size), next_action]

                td_error = (next_reward + reward_noise[:,:,i]) + np.multiply(next_terminal*self.gamma,next_pred) - current_pred
                self.update_vec[:] = data_state.T.dot(td_error).squeeze()
                self.wvecs[i] += ((self.alpha/batch_size)*self.update_vec)


def init(params):
    return BootDQN(params)

def get_params():
    return ["alpha","p"]
