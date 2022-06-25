import pickle
import random
import torch
import numpy as np
from rl_utils import ReplayBuffer

with open('data/buffer.bin', 'rb') as f:
    buffer = pickle.load(f)

print(type(buffer))
a, b, c, d, _ = buffer.sample(10)
print(int(buffer.size()/64))