import gym
import numpy as np
import torch
import pickle
import random
from rl_utils import ReplayBuffer

with open('data/buffer.bin', 'rb') as f:
    buffer = pickle.load(f)

'''
    生成器的输入是噪声（正态分布随机数）输出要和样本相同
    但是目的是找一个状态，动作为输入，输出为下一状态的生成器
    噪声维度固定为|S|+|A|
'''
lr = 1e-4

class State_Generator(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(State_Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, state_dim)
        )

    def forward(self, x, a):
        nos = torch.cat([x, a], dim=1)
        x = self.main(nos)
        return x


# 判别器的输入是生成器的样本和真实样本 输出是二分类的概率值，输出使用sigmoid
# BCELoss计算交叉熵损失
class Discriminator(torch.nn.Module):
    def __init__(self, state_dim):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.LeakyReLU(),  # 在负值部分保留一定的梯度
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        prob = self.main(x)
        return prob


ge_model = State_Generator(4, 1)
dis_model = Discriminator(4)
g_optimizer = torch.optim.Adam(ge_model.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(dis_model.parameters(), lr=lr)

D_loss = []
G_loss = []
loss_func = torch.nn.BCELoss()


def train(batch_size):
    for epoch in range(30):
        d_epoch_loss = 0
        g_epoch_loss = 0
        for i in range(int(buffer.size() / batch_size)):
            b_s, b_a, b_r, b_ns, b_d = buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                               'dones': b_d}
            real_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
            states = torch.tensor(transition_dict['states'], dtype=torch.float)
            actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1)
            dis_model.zero_grad()
            real_output = dis_model(real_states)
            d_real_loss = loss_func(real_output, torch.ones_like(real_output))
            d_real_loss.backward()
            fake_states = ge_model(states, actions)
            fake_output = dis_model(fake_states.detach())
            d_fake_loss = loss_func(fake_output, torch.zeros_like(fake_output))
            d_fake_loss.backward()

            d_loss = d_real_loss + d_fake_loss
            d_optimizer.step()

            ge_model.zero_grad()
            fake_output = dis_model(fake_states)
            g_loss = loss_func(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            g_optimizer.step()

            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss
        with torch.no_grad():
            d_epoch_loss /= int(buffer.size() / batch_size)
            g_epoch_loss /= int(buffer.size() / batch_size)
            D_loss.append(d_epoch_loss)
            G_loss.append(g_epoch_loss)


if __name__ == '__main__':
    train(64)
    print(G_loss)
    print(D_loss)
