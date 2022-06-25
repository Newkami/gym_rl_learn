import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from rl_utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

with open('data/buffer.bin', 'rb') as f:
    buffer = pickle.load(f)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")
batch_size = 64
gpu = 0
LAMBDA = 10
CRITIC_ITER = 5


class State_Generator(nn.Module):
    def __init__(self, state_dim, action):
        super(State_Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, state_dim)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, state_dim):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.LeakyReLU(),  # 在负值部分保留一定的梯度
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        score = self.main(x)
        return score


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)  # 对真实数据和虚假数据进行线性插值

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)  # 用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def get_torch_variable(arg):
    if use_cuda:
        return Variable(arg).cuda(gpu)
    else:
        return Variable(arg)


gen = State_Generator(4, 1).to(device)
dis = Discriminator(4).to(device)
print(gen)
print(dis)

optimizerD = optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))


def train(batch_size, epoch):
    writer = SummaryWriter(log_dir="./runs/result1")
    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        one = one.to(device)
        mone = mone.to(device)
    for g_iter in range(epoch):
        d_iter_loss = 0
        g_iter_loss = 0
        WD_iter_loss = 0
        count = int(buffer.size() / batch_size)
        print("epoch:", g_iter)
        for i in range(int(buffer.size() / batch_size)):
            for p in dis.parameters():
                p.requires_grad = True
            # 获取样本，并进行简单张量化处理
            b_s, b_a, b_r, b_ns, b_d = buffer.sample(batch_size)
            b_s, b_ns = np.round(b_s, 4), np.round(b_ns, 4)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                               'dones': b_d}

            real_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
            states = torch.tensor(transition_dict['states'], dtype=torch.float)
            actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1)
            real_states, states, actions = get_torch_variable(real_states), get_torch_variable(
                states), get_torch_variable(actions)

            for d_iter in range(CRITIC_ITER):
                optimizerD.zero_grad()

                d_loss_real = dis(real_states)
                d_loss_real = d_loss_real.mean()
                d_loss_real = d_loss_real.unsqueeze(0)
                d_loss_real.backward(mone)

                fake_states = gen(states, actions)
                d_loss_fake = dis(fake_states)
                d_loss_fake = d_loss_fake.mean().unsqueeze(0)
                d_loss_fake.backward(one)
                # print("batch:", i, "CRITIC_ITER:", d_iter, d_loss_real.data, d_loss_fake.data)
                gradient_penalty = calc_gradient_penalty(dis, real_states.data, fake_states.data)
                gradient_penalty.backward()

                D_cost = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
            optimizerD.step()
            for p in dis.parameters():
                p.requires_grad = False
            gen.zero_grad()
            fake_states = gen(states, actions)
            G = dis(fake_states)
            G = G.mean().unsqueeze(0)
            G.backward(mone)
            G_cost = -G
            optimizerG.step()

            with torch.no_grad():
                d_iter_loss += D_cost
                g_iter_loss += G_cost
                WD_iter_loss += Wasserstein_D
        with torch.no_grad():
            d_iter_loss /= count
            g_iter_loss /= count
            WD_iter_loss /= count
            writer.add_scalar(tag="d_iter_loss", scalar_value=d_iter_loss, global_step=g_iter)
            writer.add_scalar(tag="g_iter_loss", scalar_value=g_iter_loss, global_step=g_iter)
            writer.add_scalar(tag="WD_iter_loss", scalar_value=WD_iter_loss, global_step=g_iter)


if __name__ == '__main__':
    train(batch_size, 2000)


