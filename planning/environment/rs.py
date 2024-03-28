import numpy as np
from utils.dummy import DummyObject
import matplotlib.pyplot as plt

class riverswim():
    def __init__(self, params):
        
        self._deterministic = True
        self.episodic = False
        self.num_action = 2
        self.obs_dim = 1

        self.state = None

        self.min = 0.0
        self.max = 1.0

        self.goal_dimension = 0.1
        self.def_displacement = 0.1

        self.a1FailLeft = 0.7
        self.a1FailMidGoLeft = 0.1
        self.a1FailMidStay = 0.7
        self.a1SuccessRight = 0.3

        # Original sigma = 0.01
        self.sigma = 0.01
        # self.sigma = 0.1
        # No noise
        # self.sigma = 0.0

        self.goal_coor = self.max - self.goal_dimension #1000#

        self.was_reset = False

        # self.np_random = params.np_random
        self.logger = params.logger

        self.np_random_start = np.random.RandomState(params.np_random_seed)
        self.np_random_reward = np.random.RandomState(params.np_random_seed)
        self.np_random_trans = np.random.RandomState(params.np_random_seed)
        self.np_random = np.random.RandomState(params.np_random_seed)

        self.print_mode = params.environment_params.print_mode
        self.print_frequency = params.environment_params.print_frequency
        self.all_start_mode = params.environment_params.all_start_mode

        params.environment_params.num_action = self.num_action
        params.environment_params.obs_dim = self.obs_dim
        params.environment_params.obs_limits = [[0.0,1.0,1.0]]

        self.time_step = 0

        # self.max_reward = None

        # self.num_trans = 100

        # self.dist_data = DummyObject()
        # self.dist_data.current_observation = []
        # self.dist_data.current_action = []
        # self.dist_data.next_observation = []
        # self.dist_data.next_reward = []
        # self.dist_data.next_terminal = []
        # succ_reward = 0
        # succ_right = 0
        # for i in np.arange(0,1,0.01):
        #     p = i
        #     current_obs = np.array([p])
        #     for k in range(self.num_action):
        #         for j in range(self.num_trans):
        #             self.set_state(current_obs)
        #             next_obs, next_reward, next_terminal, _ = self.step(k)
        #             if next_reward == 1.0:
        #                 succ_reward += 1
        #             if k==1 and current_obs[0] < next_obs[0]:
        #                 succ_right += 1
        #             self.dist_data.current_observation.append(current_obs)
        #             self.dist_data.current_action.append(k)
        #             self.dist_data.next_observation.append(next_obs)
        #             self.dist_data.next_reward.append(next_reward)
        #             self.dist_data.next_terminal.append(next_terminal)
        # print(succ_reward,succ_right,len(self.dist_data.current_observation))


    def set_state(self, state):
        self.state = np.zeros(1)
        self.state[:] = state

    def reset(self):
        self.was_reset = False
        return self.internal_reset()

    def internal_reset(self):
        if not self.was_reset:
            self.state = np.zeros(1)

            if self.all_start_mode:
                self.state[0] = self.np_random.uniform(low=0, high=1.0)
            else:
                self.state[0] = self.np_random.uniform(low=0, high=0.1)

            self.was_reset = True

        return self._get_ob()

    def _get_ob(self):
        s = np.copy(self.state)
        # reverse observation
        return (1.0-s)

    def _reward(self,pos):
        if (pos > (self.max - self.goal_dimension)):
          reward = 1.
        elif (pos <= (self.min + self.goal_dimension)):
          reward = (5.0/1000)
        else:
          reward = 0.
        if not self._deterministic:
          reward+=self.np_random.randn()
        return reward
    def step(self,a):
        if self.print_mode and self.time_step%self.print_frequency == 0:
            self.logger.info("Env:{},{},{}".format(str(self.time_step), str(self.state), str(a)))

        s = self.state

        pos = s[0]

        n = self.np_random_trans.normal(scale=self.sigma)

        temp_displacement = self.def_displacement

        # if a == 1:
        #     # fails to swim upstream
        #     if (self.np_random.uniform(low=0, high=1.0) <= 0.3):
        #         temp_displacement = 0.0
        #         n = 0.0
        #     # slips and swims downstream
        #     if (self.np_random.uniform(low=0, high=1.0) <= 0.1):
        #         a = 0

        # if in 1st bin
        if (pos <= self.min + self.goal_dimension):
            #swimming right
            if(a == 1):
                #stay same place with p = 0.7
                if (self.np_random.uniform(low=0, high=1.0) < self.a1FailLeft):
                    temp_displacement = 0.0

            #swimming left
            else:
                #stay same place with p = 1.0
                temp_displacement = 0.0
        #if in last bin
        elif (pos > self.max - self.goal_dimension):
            #swimming right
            if (a == 1):
                #stay same place with p = 0.3
                if (self.np_random.uniform(low=0, high=1.0) < self.a1SuccessRight):
                    temp_displacement = 0.0
                #swim left
                else:
                    a = 0
        #in intermediate bin
        else:
            #swimming right
            if (a == 1):
                temp = self.np_random.uniform(low=0, high=1.0)
                #swim left with p = 0.1
                if(temp < self.a1FailMidGoLeft):
                    a = 0
                #stay in same place with p = 0.6
                elif(temp < self.a1FailMidStay):
                    temp_displacement = 0.0

        if a == 0:
            pos -= (temp_displacement+n)
        else:
            pos += (temp_displacement+n)

        if pos > self.max:
            pos = self.max
        elif pos < self.min:
            pos = self.min

        s[0] = pos
        self.state = s

        terminal = False
        reward = self._reward(pos)

        # if self.max_reward is None:
        #     self.max_reward = (2*reward)
        # elif reward > self.max_reward:
        #     self.max_reward = (2*reward)
        #
        # reward -= self.max_reward

        # if terminal:
        #     if self.print_mode:
        #         print("Reached terminal")
        #     self.reset()

        self.time_step += 1

        # if(self.time_step > 1 and self.time_step % 1 == 0):
        #     self._render()

        return (self._get_ob(), reward, terminal, {})

    
    def _render(self, mode='human', close=False):
        fig, ax = plt.subplots()
        ax.set_ylim([0, 1])
        y_ticks = [0, 0.5, 1]
        ax.set_yticks(y_ticks)

        ax.set_xlim([0, 1])
        x_ticks = [0, 0.5, 1]
        ax.set_xticks(x_ticks)

        ax.plot(self.state, 0, marker="o", markersize=10)
        plt.pause(0.001)
        plt.close()

    @staticmethod
    def state_dim():
        return 1

    @staticmethod
    def num_action():
        return 2


def init(params):
    return riverswim(params)