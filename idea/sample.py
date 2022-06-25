import gym
import numpy as np
import torch
import collections
import random
import pickle

# -i https://pypi.tuna.tsinghua.edu.cn/simple

buffer_size = 10000
env_name = 'CartPole-v0'


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

buffer = ReplayBuffer(buffer_size)
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.reset()
torch.manual_seed(0)

while buffer.size() < buffer_size:
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state

b_s, b_a, b_r, b_ns, b_d = buffer.sample(10000)
transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                   'dones': b_d}

with open ('data/buffer.bin','wb') as f:
     pickle.dump(buffer, f)
