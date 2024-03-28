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
import sonnet as snt
import tensorflow as tf
import tree

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
    self._prior_mean = params.agent_params.prior_mean
    self._prior_scale = params.agent_params.prior_scale
    self._num_ensemble = params.agent_params.num_heads

    # Agent hyperparameters.
    self._num_actions = params.environment.num_action
    self._batch_size = params.agent_params.batch_size
    self._sgd_period = params.agent_params.sgd_period
    self._target_update_period = params.agent_params.target_update_period
    self._min_replay_size = params.agent_params.batch_size
    self._epsilon_fn = epsilon_fn
    self._mask_prob = params.agent_params.mask_prob
    if self._num_ensemble > 1:
      self._mask_prob = params.agent_params.mask_prob
    else:
      self._mask_prob = 1.0
    self._noise_scale = params.agent_params.noise_scale
    self._rng = params.np_random
    self._rnd_seed = params.np_random_seed
    self._discount = params.agent_params.gamma
    self.learning_rate = params.agent_params.alpha
    self.optimizer_type = params.agent_params.optimizer_type

    # Agent state.
    self._total_steps = tf.Variable(1)
    self._active_head = 0

    # Agent components.
    self._ensemble = make_ensemble(
      num_actions=self._num_actions,
      input_dims = self.feature_dim,
      num_ensemble=self._num_ensemble,
      prior_mean=self._prior_mean,
      prior_scale=self._prior_scale,
      rnd_seed=self._rnd_seed)
      
    self._forward = [tf.function(net) for net in self._ensemble]
    self._target_ensemble = [copy.deepcopy(network) for network in self._ensemble]

    if self.optimizer_type == "adam":
      self._optimizer = snt.optimizers.Adam(learning_rate=self.learning_rate)
    elif self.optimizer_type == "adam_without_mom":
      self._optimizer = snt.optimizers.Adam(learning_rate=self.learning_rate, beta1=0, beta2=0.999)
    elif self.optimizer_type == "rmsprop":
      self._optimizer = snt.optimizers.RMSProp(learning_rate=self.learning_rate)
    elif self.optimizer_type == "rmsprop_with_mom":
      self._optimizer = snt.optimizers.RMSProp(learning_rate=self.learning_rate, momentum=0.9)
    elif self.optimizer_type == "sgd":
      self._optimizer = snt.optimizers.SGD(learning_rate=self.learning_rate)
    elif self.optimizer_type == "sgd_with_mom":
      self._optimizer = snt.optimizers.Momentum(learning_rate=self.learning_rate, momentum=0.999)


    self._replay = Replay(capacity=params.agent_params.buffer_size, _rng=self._rng)
    self.eval_rng = np.random.RandomState(self._rnd_seed)

    # Create variables for each network in the ensemble
    for network in self._ensemble:
      snt.build(network, (None, *(self.feature_dim,)))

    
  @tf.function
  def _step(self, transitions: Sequence[tf.Tensor]):
    """Does a step of SGD for the whole ensemble over `transitions`."""
    o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions

    o_tm1 = tf.cast(o_tm1, tf.float32)
    a_tm1 = tf.cast(a_tm1, tf.int32)
    r_t = tf.cast(r_t, tf.float32)
    d_t = tf.cast(d_t, tf.float32)
    o_t = tf.cast(o_t, tf.float32)
    m_t = tf.cast(m_t, tf.float32)
    z_t = tf.cast(z_t, tf.float32)

    variables = tree.flatten(
        [model.trainable_variables for model in self._ensemble])
    with tf.GradientTape() as tape:
      losses = []
      for k in range(self._num_ensemble):
        net = self._ensemble[k]
        target_net = self._target_ensemble[k]

        # Q-learning loss with added reward noise + half-in bootstrap.
        q_values = net(o_tm1)
        one_hot_actions = tf.one_hot(a_tm1, depth=self._num_actions)
        train_value = tf.reduce_sum(q_values * one_hot_actions, axis=-1)
        target_value = tf.stop_gradient(tf.reduce_max(target_net(o_t), axis=-1))
        target_y = r_t + z_t[:, k] + self._discount * d_t * target_value
        loss = tf.square(train_value - target_y) * m_t[:, k]
        losses.append(loss)

      loss = tf.reduce_mean(tf.stack(losses))
      gradients = tape.gradient(loss, variables)
    self._total_steps.assign_add(1)
    self._optimizer.apply(gradients, variables)

    # Periodically update the target network.
    if tf.math.mod(self._total_steps, self._target_update_period) == 0:
      for k in range(self._num_ensemble):
        for src, dest in zip(self._ensemble[k].variables,
                             self._target_ensemble[k].variables):
          dest.assign(src)

  def select_action(self, observation):
    """Select values via Thompson sampling, then use epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon_fn(self._total_steps.numpy()):
      return self._rng.randint(self._num_actions)
    
    # Before training begins, take random actions to fill the replay buffer.
    if self._replay.size < self._min_replay_size:
      action = self._rng.randint(self._num_actions)
    # Greedy policy, breaking ties uniformly at random.
    else:
      observation = tf.convert_to_tensor(observation, dtype=tf.float32)
      batched_obs = tf.expand_dims(observation, axis=0)
      q_values = self._forward[self._active_head](batched_obs)[0].numpy()
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
            r_t=copy.deepcopy(np.float32(reward)),
            d_t=copy.deepcopy(np.float32(np.logical_not(terminal))),
            o_t=copy.deepcopy(self.features),
            m_t=self._rng.binomial(1, self._mask_prob,
                                   self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
            self._noise_scale,
        ))

    self.current_observation = copy.deepcopy(self.features)
    self.current_action = copy.deepcopy(next_act)
    self.time_step += 1

    if self._replay.size < self._min_replay_size:
      return next_act

    if tf.math.mod(self._total_steps, self._sgd_period) == 0:
      minibatch = self._replay.sample(self._batch_size)
      minibatch = [tf.convert_to_tensor(x) for x in minibatch]
      self._step(minibatch)
    
    return next_act

  def eval_step(self, state, reward=None, terminal=None, step=0):
    if(step == 0) or terminal:
      self.eval_active_head = self.eval_rng.randint(self._num_ensemble)
      
    get_features(state, self.feature_constructor, self.features, self.features)
    observation = tf.convert_to_tensor(copy.deepcopy(self.features), dtype=tf.float32)
    batched_obs = tf.expand_dims(observation, axis=0)
    q_values = self._forward[self.eval_active_head](batched_obs)[0].numpy()
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)
  
  def save_policy(self, path):
    print(f"Not saving for now! ")
    # for i in range(self._num_ensemble):
    #   self._ensemble[i]._network.save_weights(str(path+"network_weights_at_step_"+str(self.time_step)+"_head_"+str(i)))
    #   self._ensemble[i]._prior_network.save_weights(str(path+"prior_network_weights_at_step_"+str(self.time_step)+"_head_"+str(i)))
      
class TransitionWithMaskAndNoise(NamedTuple):
  o_tm1: np.ndarray
  a_tm1: int
  r_t: float
  d_t: float
  o_t: np.ndarray
  m_t: np.ndarray
  z_t: np.ndarray


class NetworkWithPrior(snt.Module):
  """Combines network with additive untrainable "prior network"."""

  def __init__(self,
               network: snt.Module,
               prior_network: snt.Module,
               prior_scale: float = 1.):
    super().__init__(name='network_with_prior')
    self._network = network
    self._prior_network = prior_network
    self._prior_scale = prior_scale

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    q_values = self._network(inputs)
    prior_q_values = self._prior_network(inputs)
    return q_values + self._prior_scale * tf.stop_gradient(prior_q_values)


def make_ensemble(num_actions: int,
                  input_dims: Sequence[int],
                  num_ensemble: int = 20,
                  prior_mean: float = 0.0,
                  prior_scale: float = 3.,
                  rnd_seed: int = 0) -> Sequence[snt.Module]:
  """Convenience function to make an ensemble from flags."""
  ensemble = []
  for _ in range(num_ensemble):
    network = snt.Sequential([
        snt.Linear(num_actions, with_bias=True, w_init=snt.initializers.TruncatedNormal(mean=prior_mean, stddev=(1.0/np.power(input_dims, 1/2)), seed=1+rnd_seed+_))
    ])
    prior_network = snt.Sequential([
        snt.Linear(num_actions, with_bias=True, w_init=snt.initializers.TruncatedNormal(mean=prior_mean, stddev=(1.0/np.power(input_dims, 1/2)), seed=2+rnd_seed+_))
    ])
    ensemble.append(NetworkWithPrior(network, prior_network, prior_scale))
  return ensemble


def init(params):
    return BootstrappedDqn(params)

def get_params():
    return ["optimizer_type", "num_heads", "alpha", "prior_mean", "prior_scale", "mask_prob"]

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