from audioop import bias
import imp
from operator import index, ne
from os import stat
import numpy as np
import torch
import torch.optim as optim
from scipy.special import softmax
import matplotlib.pyplot as plt

from ..utils.agent_utils import *
from utils.replay_buffer import SimpleReplayBuffer
from utils.PlotDeepSea import PlotDeepSea
import copy

class DDQN():

    def __init__(self,
                input_size,
                output_size,
                gamma,
                learning_rate,
                np_random = np.random.RandomState(0),
                epsilon_init = 1.0,
                epsilon_final = 0.01,
                epsilon_decay_steps = 1000000,
                target_update_freq = 10000,
                update_network_freq = 4,
                num_updates = 1,
                mini_batch_size = 64,
                device = torch.device("cpu")
                ):

        # Replay buffer
        self.replay_buffer_train = SimpleReplayBuffer(np_random, max_buffer_length=100000)

        self.num_action = output_size
        self.gamma = gamma
        self.alpha = learning_rate
        self.np_random = np_random
        # update frequencies
        self.target_update_freq = target_update_freq
        self.update_network_freq = update_network_freq
        self.num_updates = num_updates
        self.mini_batch_size = mini_batch_size
        self.device = device
        
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_rate = (self.epsilon_init - self.epsilon_final) / self.epsilon_decay_steps
        self.epsilon = self.epsilon_init

        self.wvec = Net(input_size, self.num_action).to(self.device)
        # Target networks
        self.target_wvec = Net(input_size, self.num_action).to(self.device)
        self.target_wvec.load_state_dict(self.wvec.state_dict())

        # For storing current state-action-features
        self.time_step = 0
        self.episode_num = 0
        self.start_training = False

        net_params = list(self.wvec.parameters())
        # Optimizer
        self.optimizer = optim.Adam(net_params, lr=self.alpha)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def start(self, state):
        next_act = self.optimistic_action(state, self.wvec)
        self.current_state = copy.deepcopy(state)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, state, reward, terminal):
        # Storing data in the replay_buffer
        self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(state), copy.deepcopy(terminal))
        next_act = self.optimistic_action(state, self.wvec)

        # Control update frequency
        if(self.time_step % self.target_update_freq == 0 and self.time_step > 0):
            self.target_wvec.load_state_dict(self.wvec.state_dict())

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            if self.time_step % self.update_network_freq == 0:
                self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(state)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        # decaying epsilon
        self.epsilon = self.epsilon_init - self.time_step * self.epsilon_decay_rate
        if self.epsilon < self.epsilon_final:
            self.epsilon = self.epsilon_final

        return next_act

    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            next_actions = np.argmax(self.wvec.forward(next_states_t).detach().cpu(), axis=-1)
            next_values = self.target_wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            loss = self.loss_fn(current_q, target_q)

            # updating weights backpropagation        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def eval_step(self, state, reward=None, terminal=None, step=0):
        with torch.no_grad():
            hidden_state = torch.Tensor().to(self.device)
            val = self.wvec.forward(hidden_state)
            value_new = val.data.cpu().numpy()
            act = get_greedy_action(value_new, self.np_random)
            return act
    
    def optimistic_action(self, state, wvec=None):
        with torch.no_grad():
            hidden_state = torch.Tensor(np.array([state])).to(self.device)
            if self.start_training:
                if self.np_random.rand() < self.epsilon:
                    value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                else:
                    val = wvec.forward(hidden_state)
                    value_new = val.data.cpu().numpy()
                    value_new = value_new[0]
            else:
                value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                
            act = get_greedy_action(value_new, self.np_random)

        return act

    def save_policy(self, path):
        torch.save(self.wvec.state_dict(), str(path+"weights_at_step_"+str(self.time_step)))

def init(params):
    return DDQN(params)

def get_params():
    return ["optimizer_type", "alpha"]

class Net(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Net, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                7 * 7 * 64,
                256),
            torch.nn.ReLU(),
            torch.nn.Linear(
                256,
                448),
            torch.nn.ReLU(),

            torch.nn.Linear(448, 448),
            torch.nn.ReLU(),
            torch.nn.Linear(448, outputSize)
        )

        # for p in self.modules():
        #     if isinstance(p, torch.nn.Conv2d):
        #         torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        #     if isinstance(p, torch.nn.Linear):
        #         torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        # for i in range(len(self.net)):
        #     if type(self.net[i]) == torch.nn.Linear:
        #         torch.nn.init.orthogonal_(self.net[i].weight, 0.01)
        #         self.net[i].bias.data.zero_()

    def forward(self, state):
        return self.net(state)
    
class RVFs(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(RVFs, self).__init__()
        
        inputSize = 4*84*84
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(inputSize, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, outputSize, bias=True),
        )
    
    def forward(self, x):
        return self.net(x)
