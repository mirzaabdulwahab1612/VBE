import numpy as np
from collections import namedtuple, deque
import pdb
import random
import gc
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical
from copy import deepcopy
from utils import global_grad_norm_
from nn_grads_proj import *
from torch.nn import init
import gc

Transition = namedtuple('Transition', ('state', 'action', 'startState'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            intrinsic='rnd',
            probScale=True,
            stateLevel=True,
            extrinsic=False,
            numAux=10,
            reduction='max',
            var=1e-2,
            lr=1,
            optFun='adam',
            tau=0,
            iterateAve=0,
            decay=0,
            initPull=0,
            projectDim=0,
            maze=False,
            onlineUpdate=False,
            useMemory=False,
            memorySize=10000,
            syncFreq=False,
            addProb=1.
            ):

        from model_rnd import CnnActorCriticNetwork, FcActorCriticNetwork, RNDModel, FcRNDModel
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.initPull = initPull
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)

        self.actorCopy = deepcopy(self.model.actor).cuda()
        self.featureCopy = deepcopy(self.model.feature).cuda()
        self.stitchedCopy = nn.Sequential(self.featureCopy, self.actorCopy).cuda()
        self.grads_fn = get_nn_grads_fn(self.stitchedCopy)

        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.decay = decay
        self.projectDim = projectDim
        self.rnd = RNDModel(input_size, output_size)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
        self.rnd = self.rnd.to(self.device)
        self.model = self.model.to(self.device)
        self.onlineUpdate = onlineUpdate

        # for rnd_sk

        self.iterateAve = iterateAve
        self.numAux = numAux
        self.reduction = reduction
        self.var = var
        self.lr = lr
        self.auxDim = sum(p.numel() for p in self.stitchedCopy.parameters()) 

        self.auxWeights = nn.Sequential(nn.BatchNorm1d(self.auxDim), nn.Linear(self.auxDim, self.numAux, bias=False)).cuda()
        self.auxWeightsAv = deepcopy(self.auxWeights).cuda()

        self.tau = tau
        self.optFun = optFun

        self.auxUpdateCount = 0
        self.init = self.auxWeights[1].weight.data
        if optFun == 'adam': self.auxOpt = optim.Adam(self.auxWeights.parameters(), lr=self.lr, weight_decay=self.decay)
        if optFun == 'sgd': self.auxOpt = optim.SGD(self.auxWeights.parameters(), lr=self.lr, weight_decay=self.decay)
        if optFun == 'rmsprop': self.auxOpt = optim.RMSprop(self.auxWeights.parameters(), lr=self.lr, weight_decay=self.decay)
        
    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
        action = self.random_choice_prob_index(action_prob)

        entropy = (np.log(action_prob) * -1 * action_prob).sum(1).mean()
        maxProb = np.max(action_prob, 1).mean()
        #print('softmax', entropy, maxProb, flush=True)
        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    def policyInterpolate(self):

        # for actor
        for i in range(len(self.actorCopy)):
            if hasattr(self.actorCopy[i], 'weight'):
                self.actorCopy[i].weight.data = self.actorCopy[i].weight.data *  (1 - self.tau) + self.model.actor[i].weight.data * self.tau

        # for features
        for i in range(len(self.featureCopy)):
            if hasattr(self.featureCopy[i], 'weight'):
                self.featureCopy[i].weight.data = self.featureCopy[i].weight.data *  (1 - self.tau) + self.model.feature[i].weight.data * self.tau

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)


    def compute_intrinsic_reward(self, next_obs, obs=None, actions=None):

        if self.intrinsic == 'rnd':
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            target_next_feature = self.rnd.target(next_obs)
            predict_next_feature = self.rnd.predictor(next_obs)
            intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        if self.intrinsic == 'acb':

            state = torch.FloatTensor(next_obs).to(self.device)
            nActions = self.model.actor[-1].out_features

            self.stitchedCopy.zero_grad()
            grads = self.grads_fn(state, n_outputs=nActions, badge=True)

            response = self.auxWeightsAv(grads) ** 2
            intrinsic_reward = torch.max(response, 1)[0]
            del grads

        gc.collect()
        torch.cuda.empty_cache()

        return intrinsic_reward.data.detach().cpu().numpy()


    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):


        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')


        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        nActions = self.model.actor[-1].out_features
        memSplit = int(nActions)
        rand_labels = torch.randn(len(s_batch) * nActions, self.numAux).cuda()
        if self.intrinsic == 'acb': self.policyInterpolate()
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # for Curiosity-driven(Random Network Distillation)
                if self.intrinsic == 'rnd':
                    predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])
                    forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
    
                    # Proportion of exp used for predictor update
                    mask = torch.rand(len(forward_loss)).to(self.device)
                    mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                    forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))


                if self.intrinsic == 'acb':

                    mask = torch.rand(self.batch_size).to(self.device)
                    mask = mask < self.update_proportion

                    # compute features
                    batchSize = sum(mask).item()

                    bs = np.random.permutation(batchSize)[:128]

                    samps = s_batch[sample_idx][mask][bs]

                    grads = self.grads_fn(samps, n_outputs=nActions, badge=True)

                    labs = np.concatenate([sample_idx[mask.cpu().numpy()]], 0)
                    rl = rand_labels[labs][bs] * self.var
                    auxLoss = F.mse_loss(self.auxWeights(grads), rl) 
                    if self.initPull >= 0: auxLoss = auxLoss + F.mse_loss(self.auxWeights[1].weight, self.init) * self.initPull
                    auxLoss.backward()
                    self.auxOpt.step()
                    self.auxOpt.zero_grad()
                    self.auxUpdateCount += 1
                        
                    lamb = 0.
                    if self.iterateAve == 1: lamb = (self.auxUpdateCount - 1) / self.auxUpdateCount
                    self.auxWeightsAv[1].weight.data = self.auxWeightsAv[1].weight.data * lamb + self.auxWeights[1].weight.data * (1 - lamb)

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_int_loss
                if self.extrinsic: critic_loss = critic_loss + critic_ext_loss
                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy 
                if self.intrinsic == 'rnd': loss = loss + forward_loss
                loss.backward()
                global_grad_norm_(list(self.model.parameters()) + list(self.rnd.predictor.parameters()))
                self.optimizer.step()

                #del grads
                del loss, policy, value_ext, value_int, log_prob, ratio, surr1, surr2, actor_loss, critic_ext_loss, critic_int_loss, critic_loss
                gc.collect()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()


