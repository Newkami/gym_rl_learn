import numpy as np
import os
from model.QMIX.rollout import RolloutWorker
from model.QMIX.agent import Agents
from model.QMIX.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = enva

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.save_path = self.args.result_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1

        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps / / self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)

            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
            win_rate, episode_reward = self.evaluate()
            print('win_rate is ', win_rate)
            self.win_rates.append(win_rate)
            self.episode_rewards.append(episode_reward)
            # self.plt(num)

    def evaluate(self):
            win_number = 0
            episode_rewards = 0
            for epoch in range(self.args.evaluate_epoch):
                _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
                episode_rewards += episode_reward
                if win_tag:
                    win_number += 1
            return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
            plt.figure()
            plt.ylim([0, 105])
            plt.cla()
            plt.subplot(2, 1, 1)
            plt.plot(range(len(self.win_rates)), self.win_rates)
            plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
            plt.ylabel('win_rates')

            plt.subplot(2, 1, 2)
            plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
            plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
            plt.ylabel('episode_rewards')

            plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
            np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
            np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
            plt.close()
