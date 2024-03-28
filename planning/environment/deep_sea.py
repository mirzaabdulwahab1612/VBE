import numpy as np
import math
from utils.dummy import DummyObject

class DeepSea():
    def __init__(self, params):

        self.episodic = True
        self.num_action = 2
        self.obs_dim = 1

        self.size = params.environment_params.grid_size
        self.pos_min = 1

        # N*N states implementation
        # self.pos_max = int(np.power(self.size,2))

        # (N*(N+1))/2 states implementation
        self.pos_max = int((self.size * (self.size+1))/2)

        self.episode_maxpos = self.pos_max - self.size

        self.state = None
        self.actions = [-1,1]

        self.pos_range = self.pos_max - self.pos_min

        self.was_reset = False

        self.print_mode = params.environment_params.print_mode
        self.print_frequency = params.environment_params.print_frequency

        # self.np_random = params.np_random
        self.logger = params.logger

        self.np_random_start = np.random.RandomState(params.np_random_seed)
        self.np_random_reward = np.random.RandomState(params.np_random_seed)
        self.np_random_trans = np.random.RandomState(params.np_random_seed)
        self.np_random = np.random.RandomState(params.np_random_seed)

        # self.action_mapping = self.np_random_trans.binomial(1, 0.5, [self.size, self.size])
        # # self.action_mapping[:,:] = 1

        params.environment_params.num_action = self.num_action
        params.environment_params.obs_dim = self.obs_dim
        params.environment_params.obs_limits = [[self.pos_min,self.pos_max,self.pos_range]]
        params.environment_params.rmax = 1.0
        params.environment_params.rmin = -0.01/self.size
        params.environment_params.start_range = [1]

        self.last_action = None
        self.row = None
        self.column = None
        # self.last_terminal_correct = None

        self.time_step = 0

        # (N*(N+1))/2 states implementation
        # mapping indexes to states
        self.row_indexes = [np.sum(np.arange(i)) for i in range(1,self.size+1)]

    def internal_reset(self):
        if not self.was_reset:
            self.state = np.zeros((1))
            self.state[0] = 1
            self.was_reset = True
            self.row = 1
            self.column = 1
        return self._get_ob()

    def reset(self):
        self.was_reset = False
        return self.internal_reset()

    def set_state(self, state):
        self.state = self.zeros((1))
        self.state[:] = state

    def _get_ob(self):
        s = self.state
        return np.array([s[0]])

    def _terminal(self):
        return bool(self.row == (self.size+1))

    def _reward(self,terminal):
        # self.last_terminal_correct = False
        if self.row == (self.size+1) and self.column == (self.size+1):
            # self.last_terminal_correct = True
            return 1.-(0.01/self.size)
        else:
            if self.last_action == 1:
                return -0.01/self.size
            else:
                return 0

        # else:
        #     if self.last_action == 1 and (self.row == self.column):
        #         return -0.01/self.size
        #     else:
        #         return 0
        

    def step(self,a):
        if self.print_mode and self.time_step%self.print_frequency == 0:
            self.logger.info("Env:{},{},{}".format(str(self.time_step), str(self.state), str(a)))

        # action_right = (a == self.action_mapping[self.row-1, self.column-1])
        # self.last_action = action_right

        action_right = a
        self.last_action = a

        if action_right:
            self.column += 1
        else:
            self.column -= 1
            if self.column < 1:
                self.column = 1
        self.row += 1

        terminal = self._terminal()
        reward = self._reward(terminal)

        if terminal:
            if self.print_mode:
                self.logger.info("Reached terminal")
                # if self.last_terminal_correct:
                #     self.logger.info("Reached the right terminal")
            self.reset()

        s = self.state

        # NxN states implementation 
        # s[0] = ((self.row-1)*self.size)+(self.column)

        # (N*(N+1))/2 states implementation
        s[0] = self.row_indexes[self.row-1]+(self.column)
        self.state = s

        self.time_step += 1

        return (self._get_ob(), reward, terminal, {})


def init(params):
    return DeepSea(params)
