from typing import Optional
import warnings
import numpy as np

class DeepSea():
  """Deep Sea environment to test for deep exploration."""

  def __init__(self, params):

    self.noisy_reward = False
    self.episodic = True
    self.num_action = 2
    self.obs_dim = 1

    self.logger = params.logger
    self._size = params.environment_params.grid_size
    self._deterministic = params.environment_params.deterministic
    self._unscaled_move_cost = 0.01
    self.np_random = np.random.RandomState(params.np_random_seed)
    self._rng = self.np_random
    self.randomize_actions = params.environment_params.randomize_actions

    if self.randomize_actions:
      self._mapping_rng = self.np_random
      self._action_mapping = self._mapping_rng.binomial(1, 0.5, [self._size, self._size])
    else:
      warnings.warn('Environment is in debug mode (randomize_actions=False).'
                    'Only randomized_actions=True is the DeepSea environment.')
      self._action_mapping = np.ones([self._size, self._size])

    if not self._deterministic:  # action 'right' only succeeds (1 - 1/N)
      optimal_no_cost = (1 - 1 / self._size) ** (self._size - 1)
    else:
      optimal_no_cost = 1.
    self._optimal_return = optimal_no_cost - self._unscaled_move_cost

    self._column = 0
    self._row = 0
    self.state = 0
    self._bad_episode = False
    self._total_bad_episodes = 0
    self._denoised_return = 0
    self.episodes = 0


    self.pos_min = 0
    # (N*(N+1))/2 states implementation
    self.pos_max = int(((self._size * (self._size+1))/2) - 1)
    self.pos_range = self.pos_max - self.pos_min

    params.environment_params.num_action = self.num_action
    params.environment_params.obs_dim = self.obs_dim
    params.environment_params.obs_limits = [[self.pos_min,self.pos_max,self.pos_range]]

    self.row_indexes = [np.sum(np.arange(i)) for i in range(1,self._size+1)]

    # print(f"Optimal return: {self._optimal_return}")


  def _get_ob(self):
    # (N*(N+1))/2 states implementation
    self.state = self.row_indexes[self._row]+(self._column)
    return np.array([self.state])

  def reset(self):
    self._row = 0
    self._column = 0
    self._bad_episode = False
    return self._get_ob()

  def step(self, action: int):
    reward = 0.
    orig_reward = 0.
    action_right = action == self._action_mapping[self._row, self._column]

    # Reward calculation
    if self._column == self._size - 1 and action_right:
      orig_reward += 1.
      reward += 1.
      self._denoised_return += 1.
    if not self._deterministic or self.noisy_reward:  # Noisy rewards on the 'end' of chain.
      if self._row == self._size - 1 and self._column in [0, self._size - 1]:
        reward += self._rng.randn()

    # Transition dynamics
    if action_right:
      if self._rng.rand() > 1 / self._size or self._deterministic:
        self._column = np.clip(self._column + 1, 0, self._size - 1)
      reward -= self._unscaled_move_cost / self._size
      orig_reward -= self._unscaled_move_cost / self._size
    else:
      if self._row == self._column:  # You were on the right path and went wrong
        self._bad_episode = True
      self._column = np.clip(self._column - 1, 0, self._size - 1)
    self._row += 1

    if self._row == self._size:
      if self._bad_episode:
        self._total_bad_episodes += 1

      self.episodes += 1
      # self.logger.info("Reached terminal, Original Reward: {} , total_bad_episodes: {}".format(orig_reward, self._total_bad_episodes))
      self.reset()
      return (self._get_ob(), reward, True, action_right, {})

    return (self._get_ob(), reward, False, action_right, {})

  def bsuite_info(self):
      return dict(size=self._size ,episode=self.episodes, total_bad_episodes=self._total_bad_episodes,
                  denoised_return=self._denoised_return)

def init(params):
    return DeepSea(params)