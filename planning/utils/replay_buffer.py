from random import random
from random import shuffle
import numpy as np

class SimpleReplayBuffer(object):
    def __init__(self, np_random, max_buffer_length=50000):
        self.buffer = []
        # important for performance in river swim environment (smaller buffer size improves performance)
        # self.max_buffer_length = 1000
        self.max_buffer_length = max_buffer_length
        self.counter = 0
        self.np_random = np_random

    def get_buffer_size(self):
        return len(self.buffer)
        
    def add_to_buffer(self, state, action, reward, next_state, terminal):
        #data must be of the form (state, action, next_state, reward, terminal)
        if(self.counter < self.max_buffer_length):
            self.buffer.append((state, action, reward, next_state, terminal))
        else:
            self.buffer[int((self.counter-1)%self.max_buffer_length)] = (state, action, reward, next_state, terminal)
        self.counter += 1

    def sample_all_shuffled(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        indexes = list(range(len(self.buffer)))
        shuffle(indexes)
        
        for i in indexes:
            transition = self.buffer[i]
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            terminals.append(transition[4])
        return states, actions, rewards, next_states, terminals
    
    def sample_all(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for i in range(len(self.buffer)):
            transition = self.buffer[i]
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            terminals.append(transition[4])
        return states, actions, rewards, next_states, terminals

    def sample_minibatch(self,minibatch_length):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for i in range(minibatch_length):
            random_int = self.np_random.randint(0, len(self.buffer)-1) 
            transition = self.buffer[random_int]
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            terminals.append(transition[4])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminals)


    def sample_batch_lastdata(self, minibatch_length):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        # add last transition to minibatch
        transition = self.buffer[int((self.counter-1)%self.max_buffer_length)]
        states.append(transition[0])
        actions.append(transition[1])
        rewards.append(transition[2])
        next_states.append(transition[3])
        terminals.append(transition[4])

        for i in range(minibatch_length - 1):
            random_int = self.np_random.randint(0, len(self.buffer)-1) 
            transition = self.buffer[random_int]
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            terminals.append(transition[4])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminals)