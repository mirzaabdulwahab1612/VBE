import gym
import cv2
import hashlib
import numpy as np
import pdb
from abc import abstractmethod
from collections import deque
from copy import copy
#import gym_super_mario_bros
#from nes_py.wrappers import JoypadSpace as  BinarySpaceToDiscreteSpaceEnv
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

#from model import *
from PIL import Image

max_step_per_episode = 4500

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironment():
    def __init__(
            self,
            env_id,
            is_render,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=False,
            p=0.25):
        super(AtariEnvironment, self).__init__()
        if(is_render):
            self.env = MaxAndSkipEnv(gym.make(env_id, render_mode='human'), is_render)
        else:
            self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        self.env_id = env_id
        self.is_render = is_render
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def step(self, action: int):
        obsAll = []
        # uniqueStates = set()

        if 'Breakout' in self.env_id:
            action += 1

        # sticky action
        if self.sticky_action:
            if np.random.rand() <= self.p:
                action = self.last_action
            self.last_action = action

        s, reward, done, info = self.env.step(action)
        # uniqueStates.add(hashlib.sha1(s).hexdigest())
        obsAll.append(s)

        if max_step_per_episode < self.steps:
            done = True

        log_reward = reward
        force_done = done

        self.history[:3, :, :] = self.history[1:, :, :]
        self.history[3, :, :] = self.pre_proc(s)

        self.rall += reward
        self.steps += 1

        if done:
            # if np.random.rand() < 0.005: np.save(open('trajectory/lol.npy', 'wb'), np.stack(obsAll))
            self.recent_rlist.append(self.rall)
            print("[Episode {}] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                self.episode, self.steps, self.rall, np.mean(self.recent_rlist),
                info.get('episode', {}).get('visited_rooms', {})), flush=True)
            obsAll = []

            self.history = self.reset()

        return (self.history[:, :, :], reward, force_done, done, log_reward, s)

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)

