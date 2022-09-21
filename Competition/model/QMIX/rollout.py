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
                avail_action, num_ballista = self.env.get_avail_army_actions(chr(agent_id + 65))  # 可用动作的索引值
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                   epsilon, num_ballista)

                # 为动作值产生one-hot

                action_onehot = []
                for i in action:
                    temp = np.zeros(301)
                    temp[i] = 1
                    action_onehot.append(temp)
                action_onehot = np.array(action_onehot)

                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, done = self.env.step(actions)

            win_tag = True if terminated else False
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
            avail_action = self.env.get_avail_agent_actions(chr(agent_id + 65))
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
