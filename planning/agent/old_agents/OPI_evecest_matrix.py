from operator import ne
from os import stat
import numpy as np
from .utils.agent_utils import *
from utils.replay_buffer import SimpleReplayBuffer
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
        self.update_freq_policy = 64
        self.num_updates = 5
        self.mini_batch_size = 16

        self.mem_size = self.feature_dim*self.num_action
        #yvec is temporal-difference vector (Xt - gamma * Xt+1)
        self.yvec = np.zeros((self.mini_batch_size, self.mem_size))
        self.values = np.zeros(self.num_action)
        self.uncertainties = np.zeros(self.num_action)
        self.inverse_uncertainties = np.zeros(self.num_action)

        if self.feature_constructor.real_mode:
            self.features = np.zeros(self.feature_dim)
        else:
            # self.features = np.zeros(self.feature_constructor.sparse_feature_size)
            self.features = np.zeros(self.feature_dim)
            self.features_vec_state = np.zeros(self.feature_dim)
        self.features_vec = np.zeros(self.mem_size)

        # weights for value-function
        self.wvec = np.zeros(self.mem_size)
        # for uncertainty estimates
        self.uvec = np.zeros(self.mem_size)
        # for e vector in the bound
        self.evec = np.zeros(self.mem_size)

        # random weight vector for computing evec
        self.rand_wvec = self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size))
        self.rvec = np.zeros(self.mem_size)

        # For prediction
        self.policy_uvec = np.zeros(self.mem_size)
        self.policy_wvec = np.zeros(self.mem_size)
        self.policy_evec = np.zeros(self.mem_size)

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

    def start(self, observation):

        # Getting state features of next S`
        get_state(observation, self.feature_constructor, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.evec, greedy=False, target_action=False)
        get_features_state(self.features_vec, self.features, next_act, self.feature_constructor.feature_dim)

        self.current_state = copy.deepcopy(self.features_vec)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act

    def learn_uncertainty_estimates(self, wvec=None, uvec=None, evec=None, num_updates=10, mini_batch_size=64):

        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, wvec, uvec, evec, greedy=self.target_greedy_wrt_value, target_action=True)
            terminals = np.invert(terminals)

            next_values = get_values(next_states, self.wvec, self.values)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, self.wvec)
            td_error = rewards + self.gamma*terminals*next_q - current_q

            next_values = get_values(next_states, self.uvec, self.uncertainties)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, self.uvec)
            td_errors_u = td_error + self.gamma*terminals*next_q - current_q

            next_values = get_values(next_states, (self.rand_wvec - self.rvec), self.inverse_uncertainties)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, (self.rand_wvec - self.rvec))
            td_error_w_rand = current_q - self.gamma*terminals*next_q

            batch_errors = states.T.dot(td_errors_u)
            batch_errors_w_rand = states.T.dot(td_error_w_rand)

            self.uvec = self.uvec + ((self.alpha/mini_batch_size)* batch_errors)
            self.rvec = self.rvec + ((self.alpha/mini_batch_size)* batch_errors_w_rand)
            self.evec = self.rvec - self.rand_wvec

    def learn_one_update(self, wvec=None, uvec=None, evec=None, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, wvec, uvec, evec, greedy=self.target_greedy_wrt_value, target_action=True)

            terminals = np.invert(terminals)
            next_values = get_values(next_states, self.wvec, self.values)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, self.wvec)
            td_error = rewards + self.gamma*terminals*next_q - current_q

            batch_errors = states.T.dot(td_error)
            self.wvec += ((self.alpha/mini_batch_size)* batch_errors)


    def step(self, observation, reward, terminal):
        # Getting state features of next S`
        get_state(observation, self.feature_constructor, self.features)
        next_state_features = copy.deepcopy(self.features)
        
        # Storing data in the replay_buffer
        if(self.np_random.uniform(0,1) < 0.5):
            self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, next_state_features, terminal)
        else:
            self.replay_buffer_test.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, next_state_features, terminal)

        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.evec, greedy=False, target_action=False, print_flag=True)

        if next_act is None:
            return next_act

        # Getting state-action features of current S, A
        get_features_state(self.features_vec, self.features, next_act, self.feature_constructor.feature_dim)
        next_state_action_features = copy.deepcopy(self.features_vec)

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.policy_uvec = copy.deepcopy(self.uvec)
            self.policy_wvec = copy.deepcopy(self.wvec)
            self.policy_evec = copy.deepcopy(self.evec)

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size and self.replay_buffer_test.get_buffer_size() > self.mini_batch_size):
            self.learn_one_update(wvec=self.policy_wvec, uvec=self.policy_uvec, evec=self.policy_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_uncertainty_estimates(wvec=self.policy_wvec, uvec=self.policy_uvec, evec=self.policy_evec, num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = next_state_action_features
        self.current_action = next_act
        self.time_step += 1

        return next_act
    
    def optimistic_action(self, state, wvec=None, uvec=None, evec=None, greedy=False, target_action=False, print_flag=False):
        if wvec is None:
            wvec = self.wvec
        if uvec is None:
            uvec = self.uvec
        if evec is None:
            evec = self.evec

        if(greedy):
            get_values(state, wvec, self.values)
            value_new = self.values
        else:
            val = get_values(state, wvec, self.values)
            unc = get_values(state, uvec, self.uncertainties)
            iunc = get_values(state, evec, self.inverse_uncertainties)

            uncertainty_bonus = self.c * np.power(np.power((unc + iunc), 2), 0.5)            
            value_new = val + uncertainty_bonus

        act = get_greedy_action(value_new, self.np_random, target_action)

        if print_flag and self.print_mode and self.time_step%self.print_frequency==0:
            for i in range(self.values.size):
                self.logger.info("Values:{},{},{}".format(str(self.time_step), str(i), str(self.values[i])))
            self.logger.info("Action:{}".format(str(act)))

        return act

    def save_policy(self, path):
        np.save(path+"_w",self.wvec)


def init(params):
    return SARSA(params)

def get_params():
    # return ["alpha"]
    return ["alpha", "p"]