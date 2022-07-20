import gym
from collections import namedtuple
import itertools
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import rl_utils
from AC import SAC

'''
    定义环境模型,由多个高斯分布的策略的集成来构建
'''

device = torch.device("cpu")


class Swish(nn.Module):
    '''Swish激活函数'''

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        nn.init.normal_(t, mean, std)
        while True:
            cond = (t < mean - 2 * std) | (t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, nn.init.normal_(torch.ones(t.shape, device=device), mean, std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, FCLayer):
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)


class FCLayer(nn.Module):
    '''
        集成之后的全连接层
        相当于自定义一个全连接层
    '''

    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, input_dim, output_dim).to(device))
        self._activation = activation
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_dim).to(device))

    def forward(self, x):
        return self._activation(torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))


class EnsembleModel(nn.Module):
    def __init__(self, state_dim, action_dim, model_alpha, ensemble_size=5, learning_rate=1e-3):
        super(EnsembleModel, self).__init__()
        self._output_dim = (state_dim + 1) * 2
        # 输出包括均值和方差，因此是状态与奖励维度之和的二倍
        self._model_alpha = model_alpha  # 模型损失函数中加权时的权重
        self._max_logvar = nn.Parameter(
            (torch.ones((1, self._output_dim // 2)).float() / 2).to(device),  # //返回除的整数部分
            requires_grad=False
        )
        self._min_logvar = nn.Parameter(
            (-torch.ones((1, self._output_dim // 2)).float() * 10).to(device),
            requires_grad=False
        )
        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size, Swish())
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, nn.Identity())
        self.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar - F.softplus(logvar - self._max_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(
                torch.mean(torch.pow(mean - labels, 2) * inverse_var, dim=-1), dim=-1
            )
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += self._model_alpha * torch.sum(self._max_logvar) - self._model_alpha * torch.sum(self._min_logvar)
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel:
    '''环境模型集成，加入精细化的训练'''

    def __init__(self, state_dim, action_dim, model_alpha=0.01, num_network=5):
        self._num_network = num_network
        self._state_dim, self._action_dim = state_dim, action_dim
        self.model = EnsembleModel(state_dim, action_dim, model_alpha, ensemble_size=num_network)
        self._epoch_since_last_update = 0

    def train(self, inputs, labels, batch_size=64, holdout_ratio=0.1, max_iter=20):
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])

class MBPO:
    def __init__(self, env, agent, fake_env, env_pool, model_pool, rollout_length, rollout_batch_size, real_ratio,
                 num_episode):
        self.env = env
        self.agent = agent
        self.fake_env = fake_env
        self.env_pool = env_pool
        self.model_pool = model_pool
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.num_episode = num_episode
