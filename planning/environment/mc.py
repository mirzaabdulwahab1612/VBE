import numpy as np
import math
import copy
from utils.dummy import DummyObject

class MountainCar():
    def __init__(self, params):

        self._deterministic = True
        self.episodic = True
        self.num_action = 3
        self.obs_dim = 2

        self.state = None
        self.actions = [-1,0,1]

        self.pos_min = -1.2
        self.pos_max = 0.6
        self.pos_range = self.pos_max - self.pos_min
        self.vel_min = -0.07
        self.vel_max = 0.07
        self.vel_range = self.vel_max - self.vel_min

        self.was_reset = False

        self.normalized = params.environment_params.normalized
        self.sparse_reward = params.environment_params.sparse_reward
        self.print_mode = params.environment_params.print_mode
        self.print_frequency = params.environment_params.print_frequency
        self.all_start_mode = params.environment_params.all_start_mode

        # self.np_random = params.np_random
        self.logger = params.logger

        self.np_random_start = np.random.RandomState(params.np_random_seed)
        self.np_random_reward = np.random.RandomState(params.np_random_seed)
        self.np_random_trans = np.random.RandomState(params.np_random_seed)
        self.np_random = np.random.RandomState(params.np_random_seed)

        params.environment_params.num_action = self.num_action
        params.environment_params.obs_dim = self.obs_dim
        params.environment_params.obs_limits = [[self.pos_min,self.pos_max,self.pos_range],[self.vel_min,self.vel_max,self.vel_range]]
        if self.sparse_reward:
            params.environment_params.rmax = 1.0
            params.environment_params.rmin = 0.0
        else:
            params.environment_params.rmax = 0.0
            params.environment_params.rmin = -1.0
        params.environment_params.start_range = [[-0.6,-0.4],[0.0,0.0]]

        self.time_step = 0
        self.sigma = 0.0

    def internal_reset(self):
        if not self.was_reset:
            self.state = np.zeros((2))

            if self.all_start_mode:
                self.state[0] = self.np_random_start.uniform(low=self.pos_min, high=self.pos_max)
                self.state[1] = self.np_random_start.uniform(low=self.vel_min, high=self.vel_max)
            else:
                self.state[0] = self.np_random_start.uniform(low=-0.6, high=-0.4)
                self.state[1] = 0.0

            self.was_reset = True
        return self._get_ob()

    def reset(self):
        self.was_reset = False
        return self.internal_reset()

    def set_state(self, state):
        self.state = self.zeros((2))
        self.state[:] = state

    def _get_ob(self):
        if self.normalized:
            s = copy.deepcopy(self.state)
            s0 = (s[0] - self.pos_min) / self.pos_range
            s1 = (s[1] - self.vel_min) / self.vel_range
            return np.array([s0, s1])
        else:
            s = copy.deepcopy(self.state)
            return np.array([s[0], s[1]])

    def _terminal(self):
        s = copy.deepcopy(self.state)
        return bool(s[0] > self.pos_max)

    def _reward(self,terminal):
        if self.sparse_reward:
            if terminal:
                reward = 1. 
            else:
                reward = 0.
        else:
            if terminal:
                reward = 0.
            reward = -1.
        if not self._deterministic:
            reward+=self.np_random.randn()
        return reward
        

    def step(self,a):
        npos = self.np_random_trans.normal(scale=self.sigma)
        nvel = self.np_random_trans.normal(scale=self.sigma)

        s = copy.deepcopy(self.state)
        # print(s, a)
        s[1] += (0.001 * self.actions[a]) + (-0.0025 * math.cos(3.0 * s[0]))
        s[1] += nvel
        if s[1] > self.vel_max:
            s[1] = self.vel_max
        elif s[1] < self.vel_min:
            s[1] = self.vel_min

        s[0] += s[1]
        s[0] += npos

        if s[0] < self.pos_min:
            s[1] = 0.0
            s[0] = self.pos_min

        self.state = s

        terminal = self._terminal()
        reward = self._reward(terminal)

        if terminal:
            self.reset()

        self.time_step += 1

        return (self._get_ob(), reward, terminal, {})


def init(params):
    return MountainCar(params)