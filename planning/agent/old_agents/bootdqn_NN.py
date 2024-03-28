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

from ..utils.agent_utils import *
from utils.dummy import DummyObject

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree


class BootstrappedDqn():
  """Bootstrapped DQN with additive prior functions."""

  def __init__(
      self,
      params,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
      seed: Optional[int] = 42,
  ):
    """Bootstrapped DQN with additive prior functions."""
    # Agent components.
    # self._ensemble = ensemble
    self._num_actions = params.environment.num_action
    self._num_ensemble = params.agent_params.num_heads
    self._prior_scale = params.agent_params.prior_scale
    self.learning_rate = params.agent_params.alpha
    self.feature_dim = params.feature_constructor.feature_dim
    self.obs_dim = params.environment.obs_dim
    self.feature_constructor = params.feature_constructor

    self._ensemble = make_ensemble(
      num_actions=self._num_actions,
      num_ensemble=self._num_ensemble,
      num_hidden_layers=2,
      num_units=50,
      prior_scale=self._prior_scale)

    self._forward = [tf.function(net) for net in self._ensemble]
    self._target_ensemble = [copy.deepcopy(network) for network in self._ensemble]
    self._optimizer = snt.optimizers.Adam(learning_rate=self.learning_rate)
    # self._optimizer = snt.optimizers.SGD(learning_rate=self.learning_rate)
    # self._optimizer = snt.optimizers.Momentum(learning_rate=self.learning_rate, momentum=0.999, use_nesterov=False)
    # self._optimizer = snt.optimizers.RMSProp(learning_rate=self.learning_rate, momentum=0.99)
    # self._replay = replay.Replay(capacity=params.agent_params.replay_capacity)

    # Create variables for each network in the ensemble
    for network in self._ensemble:
      snt.build(network, (None, *(self.feature_dim,)))

    print(f"self._ensemble: {self._ensemble[0]._network.trainable_variables}")

    # Agent hyperparameters.
    self._batch_size = params.agent_params.batch_size
    self._sgd_period = params.agent_params.sgd_period
    self._target_update_period = params.agent_params.target_update_period
    # self.policy_update_frequency  = params.environment_params.grid_size
    self.policy_update_frequency  = 10000
    self._min_replay_size = params.agent_params.min_replay_size
    self._epsilon_fn = epsilon_fn
    self._mask_prob = params.agent_params.mask_prob
    self._noise_scale = params.agent_params.noise_scale
    self._rng = params.np_random
    self._discount = params.agent_params.gamma

    # Agent state.
    self._total_steps = tf.Variable(1)
    tf.random.set_seed(seed)

    self.np_random = params.np_random
    self.current_head = 0
    self.buffer_size = params.agent_params.buffer_size
    self.buffer_full = False
    self.nonlinear_rep = params.agent_params.nonlinear_rep
    self.weight_reward = params.agent_params.weight_reward
    self.features = np.zeros(self.feature_dim)
    self.current_data = DummyObject()
    self.current_data.current_observation = np.zeros((self.buffer_size,self.obs_dim))
    self.current_data.current_action = np.zeros(self.buffer_size)
    self.current_data.next_observation = np.zeros((self.buffer_size,self.obs_dim))
    self.current_data.next_reward = np.zeros(self.buffer_size)
    self.current_data.next_terminal = np.zeros(self.buffer_size, dtype=bool)
    if not self.nonlinear_rep:
      self.current_data.current_state_representation = np.zeros((self.buffer_size,self.feature_dim))
      self.current_data.next_state_representation = np.zeros((self.buffer_size,self.feature_dim))
    self.current_data.flags = np.zeros((self.buffer_size,self._num_ensemble))
    self.current_data.reward_noise = np.zeros((self.buffer_size,self._num_ensemble))
    
    self.current_pos = 0
    self.time_step = 0

    print(f"BOOTDQN_NN")


  def start(self, observation):
    # get_state(observation, self.feature_constructor, self.features)
    get_features(observation, self.feature_constructor, self.features, self.features)
    self.observation = observation
    next_act = self.select_action(self.features)

    self.current_data.current_observation[self.current_pos,:] = observation
    self.current_data.current_action[self.current_pos] = next_act
    if not self.nonlinear_rep:
      self.current_data.current_state_representation[self.current_pos,:] = np.copy(self.features)
    self.current_data.flags[self.current_pos,:] = self.np_random.binomial(1, self._mask_prob, self._num_ensemble)
    
    self.current_observation = observation
    self.current_action = next_act
    self.time_step += 1

    return next_act


  def step(self, observation, reward, terminal):
    # print(f"Step")

    self.current_data.next_observation[self.current_pos,:] = observation
    self.current_data.next_reward[self.current_pos] = reward
    self.current_data.next_terminal[self.current_pos] = terminal
    # get_state(observation, self.feature_constructor, self.features)
    get_features(observation, self.feature_constructor, self.features, self.features)

    if not self.nonlinear_rep:
      self.current_data.next_state_representation[self.current_pos,:] = np.copy(self.features)

    self.update_weights()

    self.current_pos += 1
    if self.current_pos == self.buffer_size:
      if not self.buffer_full:
        self.buffer_full = True
      self.current_pos = 0

    self.observation = observation
    if (self.policy_update_frequency != -1 and self.time_step%self.policy_update_frequency == 0) or terminal:
      # print(f"Updating policy head at: {self.time_step} terminal: {terminal} current observation: {observation}")
      self.current_head = self.np_random.choice(self._num_ensemble,1)[0]
    next_act = self.select_action(self.features)

    self.current_data.current_observation[self.current_pos,:] = observation
    self.current_data.current_action[self.current_pos] = next_act
    if not self.nonlinear_rep:
      self.current_data.current_state_representation[self.current_pos,:] = np.copy(self.features)
    self.current_data.flags[self.current_pos,:] = self.np_random.binomial(1, self._mask_prob, self._num_ensemble)

    # Reward noise
    reward_noise = self.np_random.randn(self._num_ensemble).astype(np.float32) * self._noise_scale
    self.current_data.reward_noise[self.current_pos,:] = reward_noise

    self.current_observation = observation
    self.current_action = next_act

    self.time_step += 1

    return next_act

  def update_weights(self):

    batch_size = self._batch_size
    if self.buffer_full:
      data_state_all = self.current_data.current_state_representation
      current_actions_all = self.current_data.current_action
      data_next_state_all = self.current_data.next_state_representation
      next_reward_all = self.current_data.next_reward
      next_terminal_all = np.logical_not(self.current_data.next_terminal)
      flags_all = self.current_data.flags
      reward_noise_all = self.current_data.reward_noise
      size = self.buffer_size
    else:
      data_state_all = self.current_data.current_state_representation[:self.current_pos+1,:]
      current_actions_all = self.current_data.current_action[:self.current_pos+1]
      data_next_state_all = self.current_data.next_state_representation[:self.current_pos+1,:]
      next_reward_all = self.current_data.next_reward[:self.current_pos+1]
      next_terminal_all = np.logical_not(self.current_data.next_terminal[:self.current_pos+1])
      flags_all = self.current_data.flags[:self.current_pos+1,:]
      reward_noise_all = self.current_data.reward_noise[:self.current_pos+1,:]
      size = self.current_pos+1

    if batch_size > size:
      batch_size = size

    indices = self.np_random.choice(size,batch_size,replace=False)
    data_state = data_state_all[indices,:]
    current_actions = current_actions_all[indices]
    data_next_state = data_next_state_all[indices,:]
    next_reward = next_reward_all[indices]
    next_terminal = next_terminal_all[indices]
    reward_noise = reward_noise_all[indices,:]
    flags = flags_all[indices,:]

    if self.current_pos < self._min_replay_size:
      return

    minibatch = (data_state, current_actions, next_reward, next_terminal, data_next_state, flags, reward_noise)
    # if tf.math.mod(self._total_steps, self._sgd_period) == 0:
    #   minibatch = (data_state, current_actions, next_reward, next_terminal, data_next_state, flags, reward_noise)
    #   minibatch = [tf.convert_to_tensor(x, dtype=tf.float32) for x in minibatch]
    self._step(minibatch)

    
  #   @tf.function
  def _step(self, transitions):
    # print("_step")
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

    # Greedy policy, breaking ties uniformly at random.
    observation = tf.convert_to_tensor(observation, dtype=tf.float32)
    batched_obs = tf.expand_dims(observation, axis=0)
    q_values = self._forward[self.current_head](batched_obs)[0].numpy()
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    # print(f"Action: {action}")
    return int(action)

  def update(
      self,
      timestep,
      action,
      new_timestep
  ):
    """Update the agent: add transition to replay and periodically do SGD."""
    if new_timestep.last():
      self._active_head = self._rng.randint(self._num_ensemble)

    self._replay.add(
        TransitionWithMaskAndNoise(
            o_tm1=timestep.observation,
            a_tm1=action,
            r_t=np.float32(new_timestep.reward),
            d_t=np.float32(new_timestep.discount),
            o_t=new_timestep.observation,
            m_t=self._rng.binomial(1, self._mask_prob,
                                   self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
            self._noise_scale,
        ))

    if self._replay.size < self._min_replay_size:
      return

    if tf.math.mod(self._total_steps, self._sgd_period) == 0:
      minibatch = self._replay.sample(self._batch_size)
      minibatch = [tf.convert_to_tensor(x) for x in minibatch]
      self._step(minibatch)


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
                  num_ensemble: int = 20,
                  num_hidden_layers: int = 2,
                  num_units: int = 50,
                  prior_scale: float = 3.) -> Sequence[snt.Module]:
  """Convenience function to make an ensemble from flags."""
  output_sizes = [num_units] * num_hidden_layers + [num_actions]
  print(f"output_sizes: {output_sizes}")
  ensemble = []
  for _ in range(num_ensemble):
    network = snt.Sequential([
        snt.nets.MLP(output_sizes),
    ])
    prior_network = snt.Sequential([
        snt.nets.MLP(output_sizes),
    ])
    ensemble.append(NetworkWithPrior(network, prior_network, prior_scale))
  return ensemble


def default_agent(
    obs_spec,
    action_spec,
    num_ensemble: int = 20,
) -> BootstrappedDqn:
  """Initialize a Bootstrapped DQN agent with default parameters."""
  ensemble = make_ensemble(
      num_actions=action_spec.num_values, num_ensemble=num_ensemble)
  optimizer = snt.optimizers.Adam(learning_rate=1e-3)
  # optimizer = snt.optimizers.SGD(learning_rate=1e-3)
  # optimizer = snt.optimizers.Momentum(learning_rate=1e-3, momentum=0.9, use_nesterov=True)
  # optimizer = snt.optimizers.RMSProp(learning_rate=1e-3)
  return BootstrappedDqn(
      obs_spec=obs_spec,
      action_spec=action_spec,
      ensemble=ensemble,
      batch_size=128,
      discount=.99,
      replay_capacity=10000,
      min_replay_size=128,
      sgd_period=1,
      target_update_period=4,
      optimizer=optimizer,
      mask_prob=0.5,
      noise_scale=0.0,
      epsilon_fn=lambda t: 10 / (10 + t),
      seed=42,
  )

def init(params):
    return BootstrappedDqn(params)

def get_params():
    return ["alpha","prior_scale", "mask_prob"]