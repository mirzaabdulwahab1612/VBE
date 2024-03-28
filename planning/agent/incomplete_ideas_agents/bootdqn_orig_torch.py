# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple implementation of Bootstrapped DQN with prior networks.

References:
1. "Deep Exploration via Bootstrapped DQN" (Osband et al., 2016)
2. "Deep Exploration via Randomized Value Functions" (Osband et al., 2017)
3. "Randomized Prior Functions for Deep RL" (Osband et al, 2018)

Links:
1. https://arxiv.org/abs/1602.04621
2. https://arxiv.org/abs/1703.07608
3. https://arxiv.org/abs/1806.03335

Notes:

- This agent is implemented with TensorFlow 2 and Sonnet 2. For installation
  instructions for these libraries, see the README.md in the parent folder.
- This implementation is potentially inefficient, as it does not parallelise
  computation across the ensemble for simplicity and readability.
"""

import copy
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.optim as optim

from ..utils.agent_utils import *

class BootstrappedDqn():
  """Bootstrapped DQN with additive prior functions."""

  def __init__(
      self,
      params,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
  ):
    """Bootstrapped DQN with additive prior functions."""

    self.policy_update_frequency = 10000
    self.time_step = 0
    self.feature_constructor = params.feature_constructor
    self.feature_dim = params.feature_constructor.feature_dim
    self.features = np.zeros(self.feature_dim)
    self._prior_scale = params.agent_params.prior_scale
    self._num_ensemble = params.agent_params.num_heads

    # Agent hyperparameters.
    self._num_actions = params.environment.num_action
    self._batch_size = params.agent_params.batch_size
    self._sgd_period = params.agent_params.sgd_period
    self._target_update_period = params.agent_params.target_update_period
    self._min_replay_size = params.agent_params.min_replay_size
    self._epsilon_fn = epsilon_fn
    self._mask_prob = params.agent_params.mask_prob
    self._noise_scale = params.agent_params.noise_scale
    self._rng = params.np_random
    self._discount = params.agent_params.gamma
    self.learning_rate = params.agent_params.alpha

    # Agent state.
    self._active_head = 0

    # Agent components.
    self._ensemble = make_ensemble(num_ensemble=self._num_ensemble, 
    inputSize=self.feature_dim, 
    outputSize=self._num_actions, 
    prior_scale=self._prior_scale)

    net_params = []
    for i in range(self._num_ensemble):
        net_params += list(self._ensemble[i]._network.parameters())
      
    self._target_ensemble = [copy.deepcopy(network) for network in self._ensemble]
    # self._optimizer = optim.SGD(net_params, lr=self.learning_rate, momentum=0.999)
    self._optimizer = optim.RMSprop(net_params, lr=self.learning_rate, alpha=0.999, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
    # self._optimizer = optim.Adam(net_params, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False)
    self._replay = Replay(capacity=params.agent_params.buffer_size, _rng=self._rng)
    self.loss_fn = torch.nn.MSELoss(reduction='mean')

  def _step(self, transitions):
    """Does a step of SGD for the whole ensemble over `transitions`."""
    o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions

    o_tm1 = torch.Tensor(o_tm1)
    r_t = torch.Tensor(r_t)
    d_t = torch.Tensor(d_t)
    o_t = torch.Tensor(o_t)
    m_t = torch.Tensor(m_t)
    z_t = torch.Tensor(z_t)

    losses = []
    for k in range(self._num_ensemble):
        net = self._ensemble[k]
        target_net = self._target_ensemble[k]

        # Q-learning loss with added reward noise + half-in bootstrap.
        q_values = net(o_tm1)
        train_value = q_values[torch.arange(len(q_values)), a_tm1]
        next_q_values = target_net(o_t).detach()
        target_value, _ = torch.max(next_q_values, dim=-1)
        target_y = r_t + z_t[:, k] + self._discount * d_t * target_value

        loss = torch.square(train_value - target_y) * m_t[:, k]
        loss = torch.mean(loss)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    # Periodically update the target network.
    if self.time_step%self._target_update_period == 0:
        for k in range(self._num_ensemble):
            self._target_ensemble[k]._network.load_state_dict(self._ensemble[k]._network.state_dict())


  def select_action(self, observation):
    """Select values via Thompson sampling, then use epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon_fn(self.time_step):
      return self._rng.randint(self._num_actions)

    observation = torch.Tensor(observation)
    # Greedy policy, breaking ties uniformly at random.
    q_values = self._ensemble[self._active_head](observation).detach().numpy()
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def start(self, observation):
    get_features(observation, self.feature_constructor, self.features, self.features)
    next_act = self.select_action(self.features)

    self.current_observation = copy.deepcopy(self.features)
    self.current_action = copy.deepcopy(next_act)
    self.time_step += 1

    return next_act
  
  def step(self, observation, reward, terminal):
    """Update the agent: add transition to replay and periodically do SGD."""
    if (self.policy_update_frequency != -1 and self.time_step%self.policy_update_frequency == 0) or terminal:
      self._active_head = self._rng.randint(self._num_ensemble)

    get_features(observation, self.feature_constructor, self.features, self.features)
    next_act = self.select_action(self.features)

    self._replay.add(
        TransitionWithMaskAndNoise(
            o_tm1=copy.deepcopy(self.current_observation),
            a_tm1=copy.deepcopy(self.current_action),
            r_t=np.float32(reward),
            d_t=np.float32(np.logical_not(terminal)),
            o_t=copy.deepcopy(self.features),
            m_t=self._rng.binomial(1, self._mask_prob,
                                   self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) * self._noise_scale,
        ))

    if self._replay.size < self._min_replay_size:
      return next_act

    if self.time_step%self._sgd_period == 0:
      minibatch = self._replay.sample(self._batch_size)
      self._step(minibatch)


    self.current_observation = copy.deepcopy(self.features)
    self.current_action = copy.deepcopy(next_act)
    self.time_step += 1
    return next_act


class TransitionWithMaskAndNoise(NamedTuple):
  o_tm1: np.ndarray
  a_tm1: int
  r_t: float
  d_t: float
  o_t: np.ndarray
  m_t: np.ndarray
  z_t: np.ndarray


class NetworkWithPrior():
  """Combines network with additive untrainable "prior network"."""

  def __init__(self,
               network,
               prior_network,
               prior_scale: float = 1.):
    super(NetworkWithPrior, self).__init__()
    self._network = network
    self._prior_network = prior_network
    self._prior_scale = prior_scale

  def __call__(self, inputs):
    q_values = self._network(inputs)
    prior_q_values = self._prior_network(inputs).detach()
    return q_values + self._prior_scale * (prior_q_values)


def make_ensemble(num_ensemble, inputSize, outputSize, prior_scale: float = 3.):
  """Convenience function to make an ensemble from flags."""
  ensemble = []
  for _ in range(num_ensemble):
    network = torch.nn.Linear(inputSize, outputSize, bias=True)
    prior_network = torch.nn.Linear(inputSize, outputSize, bias=True)
    ensemble.append(NetworkWithPrior(network, prior_network, prior_scale))
  return ensemble


def init(params):
    return BootstrappedDqn(params)

def get_params():
    return ["alpha","prior_scale", "mask_prob"]

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple, uniformly sampled replay buffer."""

from typing import Any, Optional, Sequence

import numpy as np


class Replay:
  """Uniform replay buffer. Allocates all required memory at initialization."""

  _data: Optional[Sequence[np.ndarray]]
  _capacity: int
  _num_added: int

  def __init__(self, capacity: int, _rng):
    """Initializes a new `Replay`.

    Args:
      capacity: The maximum number of items allowed in the replay. Adding
        items to a replay that is at maximum capacity will overwrite the oldest
        items.
    """
    self._data = None
    self._capacity = capacity
    self._num_added = 0
    self.rnd_seed = _rng

  def add(self, items: Sequence[Any]):
    """Adds a single sequence of items to the replay.

    Args:
      items: Sequence of items to add. Does not handle batched or nested items.
    """
    if self._data is None:
      self._preallocate(items)

    for slot, item in zip(self._data, items):
      slot[self._num_added % self._capacity] = item

    self._num_added += 1

  def sample(self, size: int) -> Sequence[np.ndarray]:
    """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
    indices = self.rnd_seed.randint(self.size, size=size)
    return [slot[indices] for slot in self._data]

  def reset(self,):
    """Resets the replay."""
    self._data = None

  @property
  def size(self) -> int:
    return min(self._capacity, self._num_added)

  @property
  def fraction_filled(self) -> float:
    return self.size / self._capacity

  def _preallocate(self, items: Sequence[Any]):
    """Assume flat structure of items."""
    as_array = []
    for item in items:
      if item is None:
        raise ValueError('Cannot store `None` objects in replay.')
      as_array.append(np.asarray(item))

    self._data = [np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape)
                  for x in as_array]

  def __repr__(self):
    return 'Replay: size={}, capacity={}, num_added={}'.format(
        self.size, self._capacity, self._num_added)