import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import random


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        print(x.size())
        print(a.size())
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x: x):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG2:
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim, discrete, action_bound, sigma, actor_lr,
                 critic_lr, tau, gamma, device):
        # self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # self.critic = QValueNet(state_dim, action_dim, hidden_dim).to(device)
        # self.target_critic = QValueNet(state_dim, action_dim, hidden_dim).to(device)

        out_fn = (lambda x: x) if discrete else (lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(num_in_actor, num_out_actor, hidden_dim,
                                activation=F.relu, out_fn=out_fn).to(device)
        self.target_actor = TwoLayerFC(num_in_actor, num_out_actor, hidden_dim,
                                       activation=F.relu, out_fn=out_fn).to(device)
        self.critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 优化器选择
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)
        # 其他各类型参数
        self.gamma = gamma
        self.sigma = sigma
        self.action_bound = action_bound
        self.tau = tau
        self.device = device
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device=self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(device=self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device=self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device=self.device)

        next_q_values = self.target_critic(torch.cat([next_states, self.target_actor(next_states)], dim=1))
        q_targets = rewards + next_q_values * self.gamma * (1 - dones)
        critic_loss = torch.mean(
            F.mse_loss(
                self.critic(torch.cat([states, actions], dim=1)), q_targets
            ))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic
                                 (torch.cat([states, self.actor(states)], dim=1))
                                 )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


actor_lr = 5e-4
critic_lr = 5e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005
buffer_size = 10000
minimai_size = 1000
batch_size = 64
sigma = 0.01
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v1'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = DDPG2(
    num_in_actor=state_dim, num_out_actor=action_dim, num_in_critic=state_dim + action_dim, hidden_dim=hidden_dim,
    discrete=False, action_bound=action_bound,
    sigma=sigma, actor_lr=actor_lr, critic_lr=critic_lr,
    tau=tau, gamma=gamma, device=device
)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimai_size, batch_size)
