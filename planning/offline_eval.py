from importlib import import_module
import numpy as np
import torch
import logging
import itertools
import datetime
from logging import handlers

from utils.parser import *
from utils.misc import *
from utils.dummy import *


def eval_policy(agent, params, eval_count):
    temp_seed = params.np_random_seed
    params.np_random_seed = 100*temp_seed+eval_count
    eval_environment = params.environment_code.init(params)
    eval_environment.print_mode = False
    params.np_random_seed = temp_seed

    step_reward = []
    if eval_environment.episodic:
        episode_reward = []
        episode_steps = []

    step = 0
    num_episodes = 0
    current_step = 0
    current_reward = 0
    cumulative_reward = 0

    observation = eval_environment.reset()
    action = agent.eval_step(observation, step=step)

    while step < params.agent_params.eval_num_steps:
        after_step = eval_environment.step(action)
        cumulative_reward += after_step[1]
        current_reward = after_step[1]
        step_reward.append(current_reward)
        step += 1
        current_step += 1

        action = agent.eval_step(after_step[0], terminal=after_step[2], step=step)
        if action is None:
            exit("Numerical error, invalid action")
        observation = after_step[0]

        if eval_environment.episodic:
            if after_step[2] or ((params.basic.max_steps_episode != -1) and (current_step == params.basic.max_steps_episode)):
                episode_reward.append(cumulative_reward)
                episode_steps.append(current_step)
                num_episodes += 1
                if not after_step[2]:
                    observation = eval_environment.reset()
                    action = agent.eval_step(observation)
                cumulative_reward = 0
                current_step = 0

        if eval_environment.episodic:
            if params.basic.num_episodes != -1 and num_episodes == params.basic.num_episodes:
                break

        if params.basic.num_steps !=-1 and step == params.basic.num_steps:
            break
    
    if eval_environment.episodic:
        # print(f"mean cumulative_reward: {np.mean(episode_reward)} episode_reward: {len(episode_reward)}")
        # return np.array(step_reward), np.array(episode_reward), np.array(episode_steps), np.mean(episode_reward), len(episode_reward)
        return np.mean(episode_reward), len(episode_reward)
    else:
        # print(f"cumulative_reward: {cumulative_reward}")
        # return np.array(step_reward), cumulative_reward
        return cumulative_reward
    