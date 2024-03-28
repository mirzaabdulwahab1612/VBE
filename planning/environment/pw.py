from dis import dis
from turtle import distance
import numpy as np


class puddle():
    def __init__(self, headX, headY, tailX, tailY, radius, length, axis):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length
        self.axis = axis

    def get_distance(self, xCoor, yCoor):

        if self.axis == 0:
            u = (xCoor - self.tailX)/self.length
        else:
            u = (yCoor - self.tailY)/self.length

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailX - xCoor),2) + np.power((self.tailY - yCoor),2))
            else:
                dist = np.sqrt(np.power((self.headX - xCoor),2) + np.power((self.headY - yCoor),2))
        else:
            x = self.tailX + u * (self.headX - self.tailX)
            y = self.tailY + u * (self.headY - self.tailY)

            dist = np.sqrt(np.power((x - xCoor),2) + np.power((y - yCoor),2))

        if dist < self.radius:
            return (self.radius - dist)
        else:
            return 0


class puddleworld():
    def __init__(self, params):

        self._deterministic = True
        self.episodic = True
        self.num_action = 4
        self.obs_dim = 2

        self.state = None
        self.puddle1 = puddle(0.45,0.75,0.1,0.75,0.1,0.35,0)
        self.puddle2 = puddle(0.45,0.8,0.45,0.4,0.1,0.4,1)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0
        self.pworld_mid_x = (self.pworld_max_x - self.pworld_min_x)/2.0
        self.pworld_mid_y = (self.pworld_max_y - self.pworld_min_y)/2.0

        self.goal_dimension = 0.05
        self.def_displacement = 0.05

        # Original sigma = 0.1
        self.sigma = 0.1
        # self.sigma = 0.01
        # No noise
        # self.sigma = 0.0

        self.goal_x_coor = self.pworld_max_x - self.goal_dimension #1000#
        self.goal_y_coor = self.pworld_max_y - self.goal_dimension #1000#

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
        params.environment_params.obs_limits = [[0.0,1.0,1.0],[0.0,1.0,1.0]]

        self.time_step = 0

    def internal_reset(self):
        if not self.was_reset:
            self.state = np.zeros(2)

            if self.all_start_mode:
                self.state[0] = self.np_random_start.uniform(low=0, high=self.goal_x_coor)
                self.state[1] = self.np_random_start.uniform(low=0, high=self.goal_y_coor)
            else:
                self.state[0] = self.np_random_start.uniform(low=0.1, high=0.3)
                self.state[1] = self.np_random_start.uniform(low=0.45, high=0.65)

            self.was_reset = True

        return self._get_ob()

    def reset(self):
        self.was_reset = False
        return self.internal_reset()

    def _get_ob(self):
        s = np.copy(self.state)
        return s

    def _terminal(self):
        s = self.state
        return bool((s[0] > self.goal_x_coor) and (s[1] > self.goal_y_coor))

    def _reward(self,x,y,terminal):
        if terminal:
            reward = -1/81.
        else:
            reward = -1.
            dist = self.puddle1.get_distance(x, y)
            reward += (-400. * dist)
            dist = self.puddle2.get_distance(x, y)
            reward += (-400. * dist)
            reward = (reward/81)
        if not self._deterministic:
            reward+=self.np_random.randn()
        return reward

    def step(self,a):

        s = self.state

        xpos = s[0]
        ypos = s[1]

        nx = self.np_random_trans.normal(scale=self.sigma)
        ny = self.np_random_trans.normal(scale=self.sigma)

        # #--type1
        # if a == 0: #up
        #     ypos += self.def_displacement
        # elif a == 1: #down
        #     ypos -= self.def_displacement
        # elif a == 2: #right
        #     xpos += self.def_displacement
        # else: #left
        #     xpos -= self.def_displacement
        #
        # xpos += nx
        # ypos += ny
        # #--type1

        # #--type2
        # if a == 0: #up
        #     ypos += (self.def_displacement+ny)
        #     xpos += nx
        # elif a == 1: #down
        #     ypos -= (self.def_displacement+ny)
        #     xpos += nx
        # elif a == 2: #right
        #     xpos += (self.def_displacement+nx)
        #     ypos += ny
        # else: #left
        #     xpos -= (self.def_displacement+nx)
        #     ypos += ny
        # #--type2

        #--type3
        if a == 0: #up
            ypos += (self.def_displacement+ny)
        elif a == 1: #down
            ypos -= (self.def_displacement+ny)
        elif a == 2: #right
            xpos += (self.def_displacement+nx)
        else: #left
            xpos -= (self.def_displacement+nx)
        #--type3

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = s

        terminal = self._terminal()
        reward = self._reward(xpos,ypos,terminal)

        if terminal:
            self.reset()

        self.time_step += 1

        # if(self.time_step > 2000 and self.time_step % 1 == 0):
        #     self._render()

        return (self._get_ob(), reward, terminal, {})

    def _render(self, mode='human', close=False):
        import matplotlib.pyplot as plt

        circle1 = plt.Circle((0.275, 0.75), 0.1, color='g')
        circle2 = plt.Circle((0.45, 0.6), 0.1, color='b')
        circle3 = plt.Circle((self.state), 0.01, color='r')

        fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
        # (or if you have an existing figure)
        # fig = plt.gcf()
        # ax = fig.gca()

        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)

        plt.pause(0.001)
        fig.clear()
        ax.clear()
        plt.close()
        
        # plt.show()


    @staticmethod
    def state_dim():
        return 2

    @staticmethod
    def num_action():
        return 4


def init(params):
    return puddleworld(params)