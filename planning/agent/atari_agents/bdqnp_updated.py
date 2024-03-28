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
import os

class BootstrappedDqn():
  """Bootstrapped DQN with additive prior functions."""

  def __init__(
      self,
      input_size,
      output_size,
      gamma,
      learning_rate,
      sgd_period = 4,
      target_update_period = 10000,
      mini_batch_size = 32,
      device = torch.device("cpu"),
      mask_prob = 0.5,
      noise_scale = 0,
      np_random_seed = 0,
      np_random = np.random.RandomState(0),
      num_heads = 20,
      prior_scale = 10.0,
      buffer_size = 100000,
      model_save_path = None,
  ):
    """Bootstrapped DQN with additive prior functions."""

    self.policy_update_frequency = 10000
    self.time_step = 0
    self._prior_scale = prior_scale
    self._num_ensemble = num_heads

    # Agent hyperparameters.
    self.device = device
    self._num_actions = output_size
    self._batch_size = mini_batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._min_replay_size = 1000
    self._mask_prob = mask_prob
    if self._num_ensemble > 1:
      self._mask_prob = mask_prob
    else:
      self._mask_prob = 1.0
    self._noise_scale = noise_scale
    self._rng = np_random
    self._rnd_seed = np_random_seed
    self._discount = gamma
    self.learning_rate = learning_rate

    # Representation network.
    self._representation = Representation().to(self.device)
    self._target_representation = copy.deepcopy(self._representation)
    # Agent state.
    self._active_head = 0
    # Agent components.
    self._ensemble = make_ensemble(
      num_actions=self._num_actions,
      input_dims = input_size,
      num_ensemble=self._num_ensemble,
      prior_scale=self._prior_scale,
      device = self.device,
      rnd_seed = self._rnd_seed)
      
    self._target_ensemble = [copy.deepcopy(network) for network in self._ensemble]

    net_params = list(self._representation.parameters())
    for i in range(self._num_ensemble):
      net_params += list(self._ensemble[i]._network.parameters())
    # Optimizer
    self.optimizer = optim.Adam(net_params, lr=self.learning_rate)
    self.loss_fn = torch.nn.MSELoss(reduction='mean')
    self._replay = Replay(capacity=buffer_size, _rng=self._rng)

    self.path = model_save_path
    self.model_save_frequency = 3000000
    if not os.path.exists(self.path):
      try:
          os.makedirs(self.path)
      except:
          assert (os.path.exists(self.path))
    
  def _step(self, transitions):
    """Does a step of SGD for the whole ensemble over `transitions`."""
    o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions

    o_tm1 = torch.Tensor(o_tm1).to(self.device)
    r_t = torch.Tensor(r_t).to(self.device)
    d_t = torch.Tensor(d_t).to(self.device)
    o_t = torch.Tensor(o_t).to(self.device)
    m_t = torch.Tensor(m_t).to(self.device)
    z_t = torch.Tensor(z_t).to(self.device)

    for k in range(self._num_ensemble):
      net = self._ensemble[k]
      target_net = self._target_ensemble[k]

      # Q-learning loss with added reward noise + half-in bootstrap.
      hidden_state = self._representation(o_tm1)
      q_values = net(hidden_state)
      train_value = q_values[torch.arange(len(q_values)), a_tm1]

      next_hidden_state = self._target_representation(o_t).detach()
      next_q_values = target_net(next_hidden_state).detach()
      target_value, _ = torch.max(next_q_values, dim=-1)
      target_y = r_t + z_t[:, k] + self._discount * d_t * target_value
      # loss = self.loss_fn(train_value, target_y) * m_t[:, k]
      loss = torch.square(train_value - target_y) * m_t[:, k]
      loss = torch.mean(loss)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def select_action(self, observation):
    with torch.no_grad():
      # Before training begins, take random actions to fill the replay buffer.
      if self._replay.size < self._min_replay_size:
        action = self._rng.randint(self._num_actions)
      # Greedy policy, breaking ties uniformly at random.
      else:
        observation = torch.Tensor(observation).to(self.device)
        observation = torch.unsqueeze(observation, 0)
        hidden_state = self._representation(observation).detach()
        q_values = self._ensemble[self._active_head](hidden_state)[0].detach().cpu().numpy()
        action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def start(self, observation):
    next_act = self.select_action(observation)
    self.current_observation = copy.deepcopy(observation)
    self.current_action = copy.deepcopy(next_act)
    self.time_step += 1

    return next_act
  
  def step(self, observation, reward, terminal):
    """Update the agent: add transition to replay and periodically do SGD."""
    if (self.policy_update_frequency != -1 and self.time_step%self.policy_update_frequency == 0) or terminal:
      self._active_head = self._rng.randint(self._num_ensemble)

    next_act = self.select_action(observation)

    self._replay.add(
        TransitionWithMaskAndNoise(
            o_tm1=copy.deepcopy(self.current_observation),
            a_tm1=copy.deepcopy(self.current_action),
            r_t=copy.deepcopy(np.float32(reward)),
            d_t=copy.deepcopy(np.float32(np.logical_not(terminal))),
            o_t=copy.deepcopy(observation),
            m_t=self._rng.binomial(1, self._mask_prob,
                                   self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
            self._noise_scale,
        ))

    self.current_observation = copy.deepcopy(observation)
    self.current_action = copy.deepcopy(next_act)
    self.time_step += 1

    if self._replay.size < self._min_replay_size:
      return next_act

    # Periodically update the target network.
    if self.time_step % self._target_update_period == 0:
      self._target_representation.load_state_dict(self._representation.state_dict())
      for k in range(self._num_ensemble):
        self._target_ensemble[k]._network.load_state_dict(self._ensemble[k]._network.state_dict())
    
    # update the network
    if self.time_step % self._sgd_period == 0:
      minibatch = self._replay.sample(self._batch_size)
      self._step(minibatch)

    # new change: save the model
    if self.time_step % self.model_save_frequency == 0:
      self.save_policy(self.path)
    
    return next_act
  
  def save_policy(self, path):
    torch.save(self._representation.state_dict(), str(path + "_representation_weights_at_step_"+str(self.time_step)))
    torch.save(self._target_representation.state_dict(), str(path + "_target_representation_weights_at_step_"+str(self.time_step)))
    torch.save(self.optimizer.state_dict(), str(path + "_optimizer_state_at_step_"+str(self.time_step)))
    for k in range(self._num_ensemble):
      torch.save(self._ensemble[k]._network.state_dict(), str(path + "_network_" + str(k)+"_weights_at_step_"+str(self.time_step)))
      torch.save(self._target_ensemble[k]._network.state_dict(), str(path + "_target_network_" + str(k)+"_weights_at_step_"+str(self.time_step)))
    

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


def make_ensemble(num_actions: int,
                  input_dims: Sequence[int],
                  num_ensemble: int = 20,
                  prior_scale: float = 3.,
                  device = torch.device("cpu"),
                  rnd_seed: int = 0 ):
  """Convenience function to make an ensemble from flags."""
  ensemble = []
  for _ in range(num_ensemble):

    network = Net(input_dims, num_actions).to(device)
    prior_network = Net(input_dims, num_actions).to(device)

    # turn off gradient for prior network
    for param in prior_network.parameters():
        param.requires_grad = False

    ensemble.append(NetworkWithPrior(network, prior_network, prior_scale))
  return ensemble

class Representation(torch.nn.Module):
  def __init__(self):
    super(Representation, self).__init__()

    self.representation = torch.nn.Sequential(
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
      torch.nn.ReLU()
    )

    for p in self.modules():
      if isinstance(p, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
        p.bias.data.zero_()

      if isinstance(p, torch.nn.Linear):
        torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
        p.bias.data.zero_()
  
  def forward(self, state):
    return self.representation(state)

class Net(torch.nn.Module):
  def __init__(self, inputSize, outputSize):
    super(Net, self).__init__()

    self.net = torch.nn.Sequential(
      torch.nn.Linear(448, 448),
      torch.nn.ReLU(),
      torch.nn.Linear(448, outputSize)
    )

    for i in range(len(self.net)):
      if type(self.net[i]) == torch.nn.Linear:
        torch.nn.init.orthogonal_(self.net[i].weight, 0.01)
        self.net[i].bias.data.zero_()

  def forward(self, state):
    return self.net(state)

def init(params):
    return BootstrappedDqn(params)

def get_params():
    return ["num_heads", "alpha", "prior_scale", "mask_prob"]

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
    # indices = [self.size - 1]
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