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
import gc

class SARSA():

    def __init__(self, params):

        # Replay buffer
        self.replay_buffer_train = SimpleReplayBuffer(params.np_random)

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
        self.features = np.zeros(self.feature_dim)

        self.device = params.device
        self.numAux = params.agent_params.numAux
        self.initPull = 1e3
        self.iterateAve = 1
        self.auxUpdateCount = 0
        self.update_proportion = 0.25
        self.tau = 1e-6
        self.num_hidden_units = params.agent_params.num_hidden_units
        
        # function approximator
        self.wvec = Net(self.feature_dim, self.num_hidden_units, self.num_action)
        self.target_wvec = Net(self.feature_dim, self.num_hidden_units, self.num_action)
        self.intrinsic_wvec = Net(self.feature_dim, self.num_hidden_units, self.num_action)
        self.target_intrinsic_wvec = Net(self.feature_dim, self.num_hidden_units, self.num_action)
        # model for features
        self.featureCopy = copy.deepcopy(self.wvec.features)
        self.valueCopy = copy.deepcopy(self.wvec.value)
        self.stitchedCopy = torch.nn.Sequential(self.featureCopy, self.valueCopy)
        self.grads_fn = get_nn_grads_fn(self.stitchedCopy)
        # ACB random network and predictors
        self.auxDim = sum(p.numel() for p in self.stitchedCopy.parameters()) 
        self.auxWeights = torch.nn.Sequential(torch.nn.BatchNorm1d(self.auxDim), torch.nn.Linear(self.auxDim, self.numAux, bias=False))
        self.auxWeightsAv = copy.deepcopy(self.auxWeights)
        self.init = self.auxWeights[1].weight.data
        # Initializing target networks
        self.target_wvec.load_state_dict(self.wvec.state_dict())
        self.target_intrinsic_wvec.load_state_dict(self.intrinsic_wvec.state_dict())
        
        # parameters for the bound
        self.target_polic_type = params.agent_params.target_polic_type
        self.reward_coeff = params.agent_params.reward_coeff
        self.bonus_coeff = params.agent_params.bonus_coeff

        # For storing current state-action-features
        self.current_state = np.zeros(self.mem_size)
        self.current_action = None
        self.time_step = 0
        self.episode_num = 0
        self.start_training = False
 
        net_params = list(self.wvec.parameters()) + list(self.intrinsic_wvec.parameters()) + list(self.auxWeights.parameters())
        # Optimizer
        self.optimizer_type = params.agent_params.optimizer_type
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(net_params, lr=self.alpha)
        elif self.optimizer_type == "adam_without_mom":
            self.optimizer = optim.Adam(net_params, lr=self.alpha, betas=(0, 0.999))
        elif self.optimizer_type == "rmsprop":
            self.optimizer = optim.rmsprop(net_params, lr=self.alpha)
        elif self.optimizer_type == "rmsprop_with_mom":
            self.optimizer = optim.rmsprop(net_params, lr=self.alpha, momentum=0.9)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(net_params, lr=self.alpha)
        elif self.optimizer_type == "sgd_with_mom":
            self.optimizer = optim.SGD(net_params, lr=self.alpha, momentum=0.9)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def start(self, observation):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        next_act = self.optimistic_action(self.features, self.wvec, self.intrinsic_wvec, target_action=False)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1
        return next_act
    
    def step(self, observation, reward, terminal):
        # Getting state features of next S`
        get_features(observation, self.feature_constructor, self.features, self.features)
        # Storing data in the replay_buffer
        self.replay_buffer_train.add_to_buffer(copy.deepcopy(self.current_state), copy.deepcopy(self.current_action), reward, copy.deepcopy(self.features), copy.deepcopy(terminal))
        next_act = self.optimistic_action(self.features, self.wvec, self.intrinsic_wvec, target_action=False, print_flag=True)

        # Control update frequency
        if(self.time_step % self.update_freq_policy == 0 and self.time_step > 0):
            self.target_wvec.load_state_dict(self.wvec.state_dict())
            self.target_intrinsic_wvec.load_state_dict(self.intrinsic_wvec.state_dict())

        if(self.replay_buffer_train.get_buffer_size() > self.mini_batch_size):
            self.start_training = True
            self.learn_one_update(num_updates=self.num_updates, mini_batch_size=self.mini_batch_size)

        self.current_state = copy.deepcopy(self.features)
        self.current_action = copy.deepcopy(next_act)
        self.time_step += 1

        return next_act
    
    def policyInterpolate(self):
        # for features
        for i in range(len(self.featureCopy)):
            if hasattr(self.featureCopy[i], 'weight'):
                self.featureCopy[i].weight.data = self.featureCopy[i].weight.data *  (1 - self.tau) + self.wvec.features[i].weight.data * self.tau

        # for value
        for i in range(len(self.valueCopy)):
            if hasattr(self.valueCopy[i], 'weight'):
                self.valueCopy[i].weight.data = self.valueCopy[i].weight.data *  (1 - self.tau) + self.wvec.value[i].weight.data * self.tau

    def compute_intrinsic_reward(self, state):
        # halucinating gradients
        self.stitchedCopy.zero_grad()
        grads = self.grads_fn(state, n_outputs=self.num_action, badge=True)
        # last layer features
        # grads = self.featureCopy(state).detach()
        response = self.auxWeightsAv(grads) ** 2
        intrinsic_reward = torch.max(response, 1)[0].data.detach()
        return intrinsic_reward
        
    def learn_one_update(self, num_updates=10, mini_batch_size=64):
        for i in range(num_updates):
            states, actions, rewards, next_states, terminals = self.replay_buffer_train.sample_batch_lastdata(mini_batch_size)

            terminals_ = torch.Tensor(np.invert(terminals)).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            next_states_t = torch.Tensor(next_states).to(self.device)
            states_t = torch.Tensor(states).to(self.device)

            if self.target_polic_type == "optimistic":
                next_actions = self.optimistic_action(next_states, self.wvec, self.intrinsic_wvec, target_action=True)
            elif self.target_polic_type == "greedy":
                next_actions = np.argmax(self.wvec.forward(next_states_t).detach(), axis=-1)
            else:
                exit("Invalid value target policy type")

            self.policyInterpolate()
            # updating extrinsic value function
            next_values = self.target_wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            target_q = rewards + self.gamma*terminals_*next_q
            extrinsic_loss = self.loss_fn(current_q, target_q)

            # updating intrinsic value function
            intrinsic_reward = self.compute_intrinsic_reward(next_states_t)
            next_values = self.target_intrinsic_wvec.forward(next_states_t).detach()
            next_q = next_values[np.arange(mini_batch_size), next_actions]
            current_values = self.intrinsic_wvec.forward(states_t)
            current_q = current_values[np.arange(mini_batch_size), actions]
            # TODO: make the intrinsic value function non-terminating like in ACB, RND, ICM paper? Done! #
            target_q = intrinsic_reward + self.gamma*next_q
            intrinsic_loss = self.loss_fn(current_q, target_q)

            # updating the auxiliary weights
            mask = torch.rand(mini_batch_size).to(self.device)
            mask = mask < self.update_proportion
            batchSize = sum(mask).item()
            bs = self.np_random.permutation(batchSize)[:mini_batch_size]
            samps = states_t[bs]
            # halucinating gradients
            grads = self.grads_fn(samps, n_outputs=self.num_action, badge=True)
            # last layer features
            # grads = self.featureCopy(samps).detach()
            rand_labels = torch.Tensor(self.np_random.randn(len(samps), self.numAux))
            rand_predictions = self.auxWeights.forward(grads)
            auxLoss = self.loss_fn(rand_predictions, rand_labels)
            if self.initPull >= 0: auxLoss = auxLoss + self.loss_fn(self.auxWeights[1].weight, self.init) * self.initPull
    
            # updating weights backpropagation
            loss = extrinsic_loss + intrinsic_loss + auxLoss      
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.auxUpdateCount += 1

            lamb = 0.
            if self.iterateAve == 1: lamb = (self.auxUpdateCount - 1) / self.auxUpdateCount
            self.auxWeightsAv[1].weight.data = self.auxWeightsAv[1].weight.data * lamb + self.auxWeights[1].weight.data * (1 - lamb)
    
    def eval_step(self, state, reward=None, terminal=None, step=0):
        with torch.no_grad():
            target_action=False
            get_features(state, self.feature_constructor, self.features, self.features)
            hidden_state = torch.Tensor(copy.deepcopy(self.features)).to(self.device)
            val = self.wvec.forward(hidden_state)
            value_new = val.data.cpu().numpy()
            act = get_greedy_action(value_new, self.np_random, target_action)
            return act
    
    def optimistic_action(self, state, wvec=None, intrinsic_wvec=None, target_action=False, print_flag=False):
        with torch.no_grad():
            hidden_state = torch.Tensor(state).to(self.device)
            if self.start_training:
                val = wvec.forward(hidden_state)
                intrinsic_val = intrinsic_wvec.forward(hidden_state)
                value_new = (self.reward_coeff*val + self.bonus_coeff*intrinsic_val).data.cpu().numpy()
                # if (not target_action):
                #     print(f"state: {np.where(state == 1)} ; val: {val} ; intrinsic_val: {intrinsic_val} ; value_new: {value_new}")
            else:
                value_new = torch.zeros((self.num_action)).data.cpu().numpy()
                
            act = get_greedy_action(value_new, self.np_random, target_action)
        return act

    def save_policy(self, path):
        torch.save(self.wvec.state_dict(), str(path+"weights_at_step_"+str(self.time_step)))

def init(params):
    return SARSA(params)

def get_params():
    return ["optimizer_type", "target_polic_type", "numAux", "reward_coeff", "bonus_coeff", "alpha"]

class Net(torch.nn.Module):
    def __init__(self, inputSize, num_hidden_units, outputSize):
        super(Net, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Linear(inputSize, num_hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_units, num_hidden_units, bias=True),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(num_hidden_units, outputSize, bias=True),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.value(x)
        return x

    def get_features(self, x):
        return self.features(x)
        

# nn_grads_proj
import math
import pdb
from torch.nn import functional as F
import types

def del_attr(obj, names):
    if len(names) == 1:
        if names[0].isnumeric():
            idx = int(names[0])
            obj[idx] = None
        else:
            delattr(obj, names[0])
    else:
        if names[0].isnumeric():
            idx = int(names[0])
            del_attr(obj[idx], names[1:])
        else:
            del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        if names[0].isnumeric():
            idx = int(names[0])
            obj[idx] = val
        else:
            setattr(obj, names[0], val)
    else:
        if names[0].isnumeric():
            idx = int(names[0])
            set_attr(obj[idx], names[1:], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)

def get_attr(obj, names):
    if len(names) == 1:
        if names[0].isnumeric():
            idx = int(names[0])
            return obj[idx]
        else:
            return getattr(obj, names[0])
    else:
        if names[0].isnumeric():
            idx = int(names[0])
            return get_attr(obj[idx], names[1:])
        else:
            return get_attr(getattr(obj, names[0]), names[1:])

def reparam(net, params, fields):
    for i,name in enumerate(fields):
        del_attr(net, name.split("."))
        set_attr(net, name.split("."), params[i])

    net.parameters = types.MethodType(lambda self: params, type(net))
    net.named_parameters = types.MethodType(lambda self: zip(fields, params), type(net))


def get_nn_grads_fn(net):
    fields = [name for name, w in net.named_parameters()]
        
    def grads_fn(x, n_outputs=None, badge=False, projections=None, offset=None, y=None):
        def f(*weights):
            if projections is None:
                reparam(net, weights, fields)
            else:
                reparam(net, [param + (weights[0] @ P.reshape(P.shape[0],-1)).reshape(P.shape[1:]) for param,P in zip(params,projections)], fields)

            nSamps = len(x)
            if badge:
                output = net(x)
                probs = F.softmax(output, 1)
                if y is None: labels = torch.argmax(output, 1).detach().long()
                elif len(y) == 0:
                    labels = torch.stack([torch.distributions.Categorical(probs[i]).sample() for i in range(nSamps)])
                else: labels = torch.Tensor(y).cuda().long()
                probs = torch.stack([probs[i, labels[i]] for i in range(nSamps)]).detach()
                ce = F.cross_entropy(output, labels, reduction='none')
                return ce * torch.sqrt(probs)
            else:

                output = net(x.repeat(n_outputs, 1, 1, 1))
                labels = torch.cat([torch.zeros(nSamps) + i for i in range(n_outputs)], 0).long().cuda()
                probs = F.softmax(output, 1)
                probs = torch.stack([probs[i, labels[i]] for i in range(len(labels))]).detach()
                ce = F.cross_entropy(output, labels, reduction='none')
                return ce * torch.sqrt(probs)
        
        params = tuple(get_attr(net, name.split(".")) for name in fields)
        if projections is None:
            jacobians = torch.autograd.functional.jacobian(f, params)
        else:
            jacobians = torch.autograd.functional.jacobian(f, (torch.zeros(projections[0].shape[0]).cuda(),))
        flat_jacobians = []
        
        if badge: out_size = len(x)
        else: out_size = len(x) * n_outputs
        for J in jacobians:
            J_ba = J.view(out_size, -1)
            flat_jacobians.append(J.view(J_ba.shape[0],-1))
        grads = torch.hstack(flat_jacobians)
        if badge: grads = grads.view(len(x), -1)
        else: grads = grads.view(n_outputs * len(x), -1)
        if offset is not None: grads = torch.cos(grads + offset)
        return grads
    
    return grads_fn