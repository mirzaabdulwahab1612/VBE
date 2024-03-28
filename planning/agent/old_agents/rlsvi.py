import numpy as np
from .utils.agent_utils import *
from utils.dummy import DummyObject
import itertools

class OptVI():

    def __init__(self, params):

        self.num_action = params.environment.num_action
        self.obs_dim = params.environment.obs_dim
        self.obs_limits = params.environment_params.obs_limits
        self.feature_dim = params.feature_constructor.feature_dim

        self.gamma = params.agent_params.gamma
        self.epsilon = params.agent_params.epsilon
        self.weight_reward = params.agent_params.weight_reward
        self.buffer_size = params.agent_params.buffer_size
        self.reg = params.agent_params.reg
        self.nonlinear_rep = params.agent_params.nonlinear_rep
        self.policy_update_frequency = params.agent_params.policy_update_frequency

        self.print_mode = params.agent_params.print_mode
        self.print_frequency = params.agent_params.print_frequency

        self.feature_constructor = params.feature_constructor

        self.np_random = params.np_random
        self.logger = params.logger

        self.mem_size = self.feature_dim*self.num_action

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
            self.current_data.current_state_only_representation = np.zeros((self.buffer_size,self.feature_dim))
            self.current_data.next_state_representation = np.zeros((self.buffer_size,self.feature_dim))
        self.current_pos = 0
        self.buffer_full = False

        if self.nonlinear_rep:
            raise Exception("Not implemented")
        else:
            self.wvec = np.zeros(self.mem_size)
            self.wvec_mean = np.zeros(self.mem_size)

        self.values = np.zeros(self.num_action)

        if self.feature_constructor.real_mode:
            self.feature_count = self.feature_dim
        else:
            self.feature_count = self.feature_constructor.sparse_feature_size
        self.features = np.zeros(self.feature_count)
        self.features_vec_state = np.zeros(self.feature_dim)
        self.features_vec = np.zeros(self.mem_size)

        self.batch_size = 32

        self.cov_mat = []
        self.target_vec = []
        for i in range(self.num_action):
            # self.cov_mat.append(np.eye(self.feature_dim))
            # self.cov_mat[-1] *= self.reg
            self.cov_mat.append(np.zeros((self.feature_dim,self.feature_dim)))
            self.target_vec.append(np.zeros(self.feature_dim))
        self.features_cov = np.zeros(self.feature_dim)


    def start(self, observation):

        if self.time_step != 0:
            self.update_weights()

        get_features(observation, self.feature_constructor, self.features, self.features_vec_state)

        next_act = self.policy_state(self.features_vec_state)
        get_features_state(self.features_vec, self.features_vec_state, next_act, self.feature_dim)

        self.current_data.current_observation[self.current_pos,:] = observation
        self.current_data.current_action[self.current_pos] = next_act
        if not self.nonlinear_rep:
            self.current_data.current_state_only_representation[self.current_pos,:] = np.copy(self.features_vec_state)

        self.current_observation = observation
        self.current_action = next_act

        self.time_step += 1
        return next_act

    def update_weights(self):

        buffer_size = self.current_pos
        if buffer_size == 0:
            buffer_size = self.buffer_size

        next_terminal = np.logical_not(self.current_data.next_terminal[:buffer_size])

        values_next = []
        for action in range(self.num_action):
            offset = int(self.feature_dim*action)
            vals = self.current_data.next_state_representation[:buffer_size,:].dot(self.wvec[offset:offset+self.feature_dim])
            values_next.append(vals[...,np.newaxis])
        values_next = np.concatenate(values_next,axis=1)
        next_action=[self.np_random.choice(np.flatnonzero(values_next[i,:] == values_next[i,:].max())) for i in range(buffer_size)]
        next_action = np.array(next_action)
        next_pred = values_next[np.arange(buffer_size), next_action]

        targets = self.current_data.next_reward[:buffer_size] + np.multiply(next_terminal*self.gamma,next_pred)

        for action in range(self.num_action):
            offset = int(self.feature_dim*action)

            # self.wvec_mean[offset:offset+self.feature_dim] = self.cov_mat[action].dot(self.target_vec[action])
            # self.wvec[offset:offset+self.feature_dim] = np.random.multivariate_normal(self.wvec_mean[offset:offset+self.feature_dim], self.cov_mat[action])

            rel_indices = np.where(self.current_data.current_action[:buffer_size]==action)[0]
            if len(rel_indices) > 0:
                self.target_vec[action][:] = self.current_data.current_state_only_representation[:buffer_size,:][rel_indices,:].T.dot(targets[rel_indices])
            else:
                self.target_vec[action][:] = 0.0
            np.fill_diagonal(self.cov_mat[action], self.cov_mat[action].diagonal() + self.reg)
            inv_mat = np.linalg.inv(self.cov_mat[action])
            self.wvec_mean[offset:offset+self.feature_dim] = inv_mat.dot(self.target_vec[action])
            self.wvec[offset:offset+self.feature_dim] = np.random.multivariate_normal(self.wvec_mean[offset:offset+self.feature_dim], inv_mat)
            np.fill_diagonal(self.cov_mat[action], self.cov_mat[action].diagonal() - self.reg)

    def step(self, observation, reward, terminal):

        if self.weight_reward:
            reward *= (1.0-self.gamma)

        # target = reward

        self.current_data.next_observation[self.current_pos,:] = observation
        self.current_data.next_reward[self.current_pos] = reward
        self.current_data.next_terminal[self.current_pos] = terminal
        get_features(observation, self.feature_constructor, self.features, self.features_vec_state)
        if not self.nonlinear_rep:
            self.current_data.next_state_representation[self.current_pos,:] = np.copy(self.features_vec_state)

        # #update covariance and bvec
        # get_features(self.current_observation, self.feature_constructor, self.features, self.features_vec_state)
        # self.cov_mat[self.current_action].dot(self.features_vec_state,out=self.features_cov)
        # temp = 1.0 + np.dot(self.features_cov,self.features_vec_state)
        # self.features_vec_state = self.features_cov/temp
        # self.cov_mat[self.current_action] -= np.outer(self.features_vec_state,self.features_cov)

        #update covariance and bvec
        get_features(self.current_observation, self.feature_constructor, self.features, self.features_vec_state)
        self.cov_mat[self.current_action] += np.outer(self.features_vec_state,self.features_vec_state)

        if not terminal:
            get_features(observation, self.feature_constructor, self.features, self.features_vec_state)
            next_act = self.policy_state(self.features_vec_state)
            # target += self.values[next_act]

        # self.target_vec[self.current_action] += (self.features_vec_state*target)

        self.current_pos += 1
        if self.current_pos == self.buffer_size:
            if not self.buffer_full:
                self.buffer_full = True
            self.current_pos = 0

        if (self.policy_update_frequency != -1 and self.time_step%self.policy_update_frequency == 0) or terminal:
            self.update_weights()
            get_features(observation, self.feature_constructor, self.features, self.features_vec_state)
            next_act = self.policy_state(self.features_vec_state)
        get_features_state(self.features_vec, self.features_vec_state, next_act, self.feature_dim)

        self.current_data.current_observation[self.current_pos,:] = observation
        self.current_data.current_action[self.current_pos] = next_act
        if not self.nonlinear_rep:
            self.current_data.current_state_only_representation[self.current_pos,:] = np.copy(self.features_vec_state)

        self.current_observation = observation
        self.current_action = next_act

        self.time_step += 1

        return next_act

    def policy_state(self, state):

        get_values_state(self.feature_constructor, state, self.wvec, self.values)

        act = get_greedy_action(self.values, self.np_random)

        if self.print_mode and self.time_step%self.print_frequency==0:
            for i in range(self.values.size):
                self.logger.info("Values:{},{},{}".format(str(self.time_step), str(i), str(self.values[i])))
            self.logger.info("Action:{}".format(str(act)))

        return act


    def save_policy(self, path):
        np.save(path+"_w",self.wvec)

    def policy_obs_eval(self, observation, epsilon=0.0, np_random=None):
        get_values_obs(observation, self.feature_constructor, self.features, self.wvec, self.values)
        act = get_action(self.values, epsilon, np_random)
        return act

def init(params):
    return OptVI(params)

def get_params():
    return ["reg"]