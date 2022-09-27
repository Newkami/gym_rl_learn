import numpy as np
import torch


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0

        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            obs = self.env.getObs()
            state = self.env.getCurrentState()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action, num_ballista = self.env.get_avail_army_actions(
                    chr(self._get_armyindex(agent_id) + 65))  # 可用动作的索引值
                is_alive = self.env.army_list[self.env._get_armyindex(agent_id)].ballista_list[agent_id % 30].is_alive
                if is_alive:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                       epsilon, num_ballista)
                else:
                    action = 0
                # 为动作值产生one-hot

                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, win = self.env.step(actions)

            # win_tag = True if terminated else False
            win_tag = win
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # last obs
        obs = self.env.getObs()
        state = self.env.getCurrentState()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action, num_ballista = self.env.get_avail_army_actions(
                chr(self._get_armyindex(agent_id) + 65))  # 可用动作的索引值
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, 301)))
            avail_u.append(np.zeros((self.n_agents, 301)))
            avail_u_next.append(np.zeros((self.n_agents, 301)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        return episode, episode_reward, win_tag, step

    def _get_armyindex(self, action_index):
        if 0 <= action_index < 30:
            army_ind = 0
        elif 30 <= action_index < 60:
            army_ind = 1
        elif 60 <= action_index < 90:
            army_ind = 2
        elif 90 <= action_index < 120:
            army_ind = 3
        elif 120 <= action_index < 150:
            army_ind = 4
        elif 150 <= action_index < 180:
            army_ind = 5
        elif 180 <= action_index < 210:
            army_ind = 6
        elif 210 <= action_index < 240:
            army_ind = 7
        elif 240 <= action_index < 270:
            army_ind = 8
        elif 270 <= action_index < 300:
            army_ind = 9

        return army_ind
