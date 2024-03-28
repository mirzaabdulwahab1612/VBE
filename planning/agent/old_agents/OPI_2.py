from operator import ne
from os import stat
import numpy as np
from .utils.agent_utils import *
from utils.replay_buffer import SimpleReplayBuffer
from scipy.special import softmax
# from utils.PlotDeepSea import PlotDeepSea
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

        if self.feature_constructor.real_mode:
            self.features = np.zeros(self.feature_dim)
        else:
            self.features = np.zeros(self.feature_dim)

        self.features_vec = np.zeros(self.mem_size)

        # weights for value-function
        # self.wvec = self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size))
        self.wvec = np.zeros(self.mem_size)
        # for uncertainty estimates
        # self.uvec = self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size))
        self.uvec = np.zeros(self.mem_size)
        
        # random target and prediction vectors
        self.num_rvecs = params.agent_params.num_rvecs
        self.rand_wvec = self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size), size=self.num_rvecs)
        self.rvec = np.zeros((self.num_rvecs, self.mem_size))
        # self.rvec = self.np_random.multivariate_normal(np.zeros(self.mem_size), np.eye(self.mem_size), size=self.num_rvecs)

        # Targets for prediction
        self.policy_wvec = copy.deepcopy(self.wvec)
        self.policy_uvec = copy.deepcopy(self.uvec)
        self.policy_rvec = copy.deepcopy(self.rvec)

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
        self.start_training = False
        print(f"OPI_evecst_matrix_multi")

    def start(self, observation):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.rvec, target_action=False)
        get_features_state(self.features_vec, self.features, next_act, self.feature_constructor.feature_dim)

        self.current_state = copy.deepcopy(self.features_vec)
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

        next_act = self.optimistic_action(self.features, self.wvec, self.uvec, self.rvec, target_action=False, print_flag=True)
        if next_act is None:
            return next_act

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.policy_wvec = copy.deepcopy(self.wvec)
            self.policy_uvec = copy.deepcopy(self.uvec)
            self.policy_rvec = copy.deepcopy(self.rvec)

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size and self.replay_buffer_test.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            # self.learn_out_uncertainty_estimates(num_updates=1, mini_batch_size=1)
            self.learn_out_uncertainty_estimates(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)
            self.learn_uncertainty_estimates(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        # Getting state-action features of current S, A
        get_features_state(self.features_vec, self.features, next_act, self.feature_constructor.feature_dim)
        self.current_state = copy.deepcopy(self.features_vec)
        self.current_action = next_act
        self.time_step += 1

        return next_act

    def learn_out_uncertainty_estimates(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_batch_lastdata(mini_batch_size)
            # states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_minibatch(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.policy_wvec, self.policy_uvec, self.policy_rvec, target_action=True)
            terminals_ = np.invert(terminals)

            # -- Slower version --
            # next_values = get_values(next_states.T, self.rand_wvec, self.num_action, multi=True)
            # next_q = next_values[np.arange(mini_batch_size), :, next_actions]
            # current_q = np.dot(states, self.rand_wvec.T)
            # terminals_ = np.tile(terminals_, (self.num_rvecs, 1)).T    
            # reward_u = current_q - self.gamma*terminals_*next_q

            # next_values = get_values(next_states.T, self.rvec, self.num_action, multi=True)
            # next_q = next_values[np.arange(mini_batch_size), :, next_actions]
            # current_q = np.dot(states, self.rvec.T)
            # td_error_w_rand = reward_u + self.gamma*terminals_*next_q - current_q
            # -- Slower version --


            # -- Faster version --
            next_values = get_values(next_states.T, (self.rand_wvec - self.rvec), self.num_action, multi=True)
            next_q = next_values[np.arange(mini_batch_size), :, next_actions]
            current_q = np.dot(states, (self.rand_wvec - self.rvec).T)
            terminals_ = np.tile(terminals_, (self.num_rvecs, 1)).T    
            td_error_w_rand = current_q - self.gamma*terminals_*next_q
            # -- Faster version --

            batch_errors_w_rand = td_error_w_rand.T.dot(states)
            self.rvec = self.rvec + ((self.alpha/mini_batch_size)* batch_errors_w_rand)
    
    def learn_uncertainty_estimates(self, num_updates=10, mini_batch_size=64):

        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_test.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.policy_wvec, self.policy_uvec, self.policy_rvec, target_action=True)
            terminals_ = np.invert(terminals)

            next_values = get_values(next_states, self.wvec, self.num_action)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, self.wvec)
            td_error = (rewards + self.gamma*terminals_*next_q) - current_q

            next_values = get_values(next_states, self.uvec, self.num_action)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, self.uvec)
            td_errors_u = (td_error + self.gamma*terminals_*next_q) - current_q

            batch_errors = states.T.dot(td_errors_u)
            self.uvec = self.uvec + ((self.alpha/mini_batch_size)* batch_errors)


    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)
            next_actions = self.optimistic_action(next_states, self.policy_wvec, self.policy_uvec, self.policy_rvec, target_action=True)

            terminals_ = np.invert(terminals)
            next_values = get_values(next_states, self.wvec, self.num_action)
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_q = np.dot(states, self.wvec)
            td_error = rewards + self.gamma*terminals_*next_q - current_q

            batch_errors = states.T.dot(td_error)
            self.wvec += ((self.alpha/mini_batch_size)* batch_errors)
    
    def optimistic_action(self, state, wvec=None, uvec=None, rvec=None, target_action=False, print_flag=False):
        if self.start_training:
            val = get_values(state, wvec, self.num_action)
            unc = get_values(state, uvec, self.num_action)

            temp_iunc = []
            rnd_target = get_values(state.T, self.rand_wvec, self.num_action, multi=True)
            rnd_pred = get_values(state.T, rvec, self.num_action, multi=True)
            rnd_error = np.abs(rnd_pred - rnd_target)
            # rnd_error = np.abs(rnd_pred - rnd_target)/np.abs(rnd_target)
            # iunc = rnd_error*rnd_target
            iunc = rnd_error

            if(target_action):
                id = np.argmax(iunc, axis=1)
                for k in range(self.num_action):
                    temp_iunc.append(iunc[np.arange(len(iunc)), id[:, k]][:, k])
                iunc = np.stack(temp_iunc, axis=1)
            else:
                id = np.argmax(iunc, axis=0)
                for k in range(self.num_action):
                    temp_iunc.append(iunc[id[k]][k])
                iunc = np.stack(temp_iunc, axis=0)

            unc = softmax(unc, axis=-1)
            iunc = softmax(iunc, axis=-1)
            uncertainty_bonus = self.c * (unc + iunc)
            # uncertainty_bonus = self.c * (iunc)
            value_new = val + uncertainty_bonus
        else:
            value_new = np.zeros((self.num_action))

        act = get_greedy_action(value_new, self.np_random, target_action)

        if print_flag and self.print_mode and self.time_step%self.print_frequency==0:
            for i in range(self.num_action):
                offset = int(i) * self.feature_dim
                # self.logger.info("Values:{},{},{}".format(str(self.time_step), str(i), str(self.wvec[offset:offset+self.feature_dim])))
                # self.logger.info("Uncertainty:{},{},{}".format(str(self.time_step), str(i), str(self.uvec[offset:offset+self.feature_dim])))
                # self.logger.info("Evec:{},{},{}".format(str(self.time_step), str(i), str(self.evec[0][offset:offset+self.feature_dim])))
                
                # self.logger.info("Values:{},{},{}".format(str(self.time_step), str(i), str(self.wvec[offset])))
                # self.logger.info("Uncertainty:{},{},{}".format(str(self.time_step), str(i), str(self.uvec[offset])))
                # self.logger.info("Evec:{},{},{}".format(str(self.time_step), str(i), str(self.evec[:, offset])))

        return act

    def save_policy(self, path):
        np.save(path+"_w",self.wvec)

def init(params):
    return SARSA(params)

def get_params():
    return ["alpha", "p"]