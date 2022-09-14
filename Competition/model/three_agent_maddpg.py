import numpy as np
import tensorflow as tf

import make_env

from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer


def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update

# actor网络定义
agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')

agent4_ddpg = MADDPG('agent4')
agent4_ddpg_target = MADDPG('agent4_target')

agent5_ddpg = MADDPG('agent5')
agent5_ddpg_target = MADDPG('agent5_target')

agent6_ddpg = MADDPG('agent6')
agent6_ddpg_target = MADDPG('agent6_target')

agent7_ddpg = MADDPG('agent7')
agent7_ddpg_target = MADDPG('agent7_target')

agent8_ddpg = MADDPG('agent8')
agent8_ddpg_target = MADDPG('agent8_target')

agent9_ddpg = MADDPG('agent9')
agent9_ddpg_target = MADDPG('agent9_target')

agent10_ddpg = MADDPG('agent10')
agent10_ddpg_target = MADDPG('agent10_target')

saver = tf.train.Saver()

# 网络初始化
agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')

agent4_actor_target_init, agent4_actor_target_update = create_init_update('agent4_actor', 'agent4_target_actor')
agent4_critic_target_init, agent4_critic_target_update = create_init_update('agent4_critic', 'agent4_target_critic')

agent5_actor_target_init, agent5_actor_target_update = create_init_update('agent5_actor', 'agent5_target_actor')
agent5_critic_target_init, agent5_critic_target_update = create_init_update('agent5_critic', 'agent5_target_critic')

agent6_actor_target_init, agent6_actor_target_update = create_init_update('agent6_actor', 'agent6_target_actor')
agent6_critic_target_init, agent6_critic_target_update = create_init_update('agent6_critic', 'agent6_target_critic')

agent7_actor_target_init, agent7_actor_target_update = create_init_update('agent7_actor', 'agent7_target_actor')
agent7_critic_target_init, agent7_critic_target_update = create_init_update('agent7_critic', 'agent7_target_critic')

agent8_actor_target_init, agent8_actor_target_update = create_init_update('agent8_actor', 'agent8_target_actor')
agent8_critic_target_init, agent8_critic_target_update = create_init_update('agent8_critic', 'agent8_target_critic')

agent9_actor_target_init, agent9_actor_target_update = create_init_update('agent9_actor', 'agent9_target_actor')
agent9_critic_target_init, agent9_critic_target_update = create_init_update('agent9_critic', 'agent9_target_critic')

agent10_actor_target_init, agent10_actor_target_update = create_init_update('agent10_actor', 'agent10_target_actor')
agent10_critic_target_init, agent10_critic_target_update = create_init_update('agent10_critic', 'agent10_target_critic')


# 通过actor网络 选出本轮动作
def get_agents_action(o_n, sess, noise_rate=0):
    agent1_action = agent1_ddpg.action(state=[o_n[0]], sess=sess) + np.random.randn(2) * noise_rate
    agent2_action = agent2_ddpg.action(state=[o_n[1]], sess=sess) + np.random.randn(2) * noise_rate
    agent3_action = agent3_ddpg.action(state=[o_n[2]], sess=sess) + np.random.randn(2) * noise_rate
    agent4_action = agent4_ddpg.action(state=[o_n[3]], sess=sess) + np.random.randn(2) * noise_rate
    agent5_action = agent5_ddpg.action(state=[o_n[4]], sess=sess) + np.random.randn(2) * noise_rate
    agent6_action = agent6_ddpg.action(state=[o_n[5]], sess=sess) + np.random.randn(2) * noise_rate
    agent7_action = agent7_ddpg.action(state=[o_n[6]], sess=sess) + np.random.randn(2) * noise_rate
    agent8_action = agent8_ddpg.action(state=[o_n[7]], sess=sess) + np.random.randn(2) * noise_rate
    agent9_action = agent9_ddpg.action(state=[o_n[8]], sess=sess) + np.random.randn(2) * noise_rate
    agent10_action = agent10_ddpg.action(state=[o_n[9]], sess=sess) + np.random.randn(2) * noise_rate
    return agent1_action, agent2_action, agent3_action, agent4_action, agent5_action, agent6_action, agent7_action, agent8_action, agent9_action, agent10_action

# actor网络的训练
def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]
    # 获取下一个情况下另外两个agent的行动
    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])


if __name__ == '__main__':
    # 环境初始化
    env = make_env.make_env('simple_tag')
    # 环境重置，获得当前环境状态【o1 , o2 , o3 .. ,o10】
    o_n = env.reset()

    # 数据可视化
    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(10)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(10)]

    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(3)]

    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(3)]

    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(3)]

    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_l1000_mean', reward_1000[i]) for i in range(3)]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init,
              agent4_actor_target_init, agent4_critic_target_init,
              agent5_actor_target_init, agent5_critic_target_init,
              agent6_actor_target_init, agent6_critic_target_init,
              agent7_actor_target_init, agent7_critic_target_init,
              agent8_actor_target_init, agent8_critic_target_init,
              agent9_actor_target_init, agent9_critic_target_init,
              agent10_actor_target_init, agent10_critic_target_init])

    summary_writer = tf.summary.FileWriter('./three_ma_summary', graph=tf.get_default_graph())


    # 十个智能体的经验回放缓冲池
    agent1_memory = ReplayBuffer(100000)
    agent2_memory = ReplayBuffer(100000)
    agent3_memory = ReplayBuffer(100000)
    agent4_memory = ReplayBuffer(100000)
    agent5_memory = ReplayBuffer(100000)
    agent6_memory = ReplayBuffer(100000)
    agent7_memory = ReplayBuffer(100000)
    agent8_memory = ReplayBuffer(100000)
    agent9_memory = ReplayBuffer(100000)
    agent10_memory = ReplayBuffer(100000)

    # e = 1
    # reward列表
    reward_100_list = [[], [], [], [] ,[],[], [], [], [] ,[]]
    for i in range(1000000):
        if i % 1000 == 0:
            # 环境重置
            o_n = env.reset()
            # 遍历智能体
            for agent_index in range(10):
                summary_writer.add_summary(sess.run(reward_1000_op[agent_index],
                                                    {reward_1000[agent_index]: np.mean(reward_100_list[agent_index])}),
                                           i // 1000)

        # 获得联合动作空间
        agent1_action, agent2_action, agent3_action, agent4_action, agent5_action, agent6_action, agent7_action, agent8_action, agent9_action, agent10_action = get_agents_action(o_n, sess, noise_rate=0)

        # 此处不需要展开，直接注释掉
        #十个agent的行动


        a = [agent1_action, agent2_action, agent3_action, agent4_action, agent5_action, agent6_action, agent7_action, agent8_action, agent9_action, agent10_action]
        # a = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]
        # #绿球的行动
        # a.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])


        # 与环境的交互
        o_n_next, r_n, d_n, i_n = env.step(a)

        # 获得每一个智能体的reward
        for agent_index in range(10):
            reward_100_list[agent_index].append(r_n[agent_index])
            reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]

        # 加入智能体对应的经验缓冲池
        agent1_memory.add(np.vstack([o_n[0], o_n[1], o_n[2],o_n[3], o_n[4], o_n[5], o_n[6], o_n[7], o_n[8], o_n[9]]),
                          np.vstack([agent1_action, agent2_action, agent3_action, agent4_action, agent5_action, agent6_action, agent7_action, agent8_action, agent9_action, agent10_action]),
                          r_n[0], np.vstack([o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[6], o_n_next[7], o_n_next[8], o_n_next[9]]), False)

        agent2_memory.add(np.vstack([o_n[1], o_n[0], o_n[2],o_n[3], o_n[4], o_n[5], o_n[6], o_n[7], o_n[8], o_n[9]]),
                          np.vstack([agent2_action, agent1_action, agent3_action, agent4_action, agent5_action, agent6_action, agent7_action, agent8_action, agent9_action, agent10_action]),
                          r_n[1], np.vstack([o_n_next[1], o_n_next[0], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[6], o_n_next[7], o_n_next[8], o_n_next[9]]), False)

        agent3_memory.add(
            np.vstack([o_n[2], o_n[0], o_n[1], o_n[3], o_n[4], o_n[5], o_n[6], o_n[7], o_n[8], o_n[9]]),
            np.vstack([agent3_action, agent1_action, agent2_action, agent4_action, agent5_action, agent6_action,
                       agent7_action, agent8_action, agent9_action, agent10_action]),
            r_n[2], np.vstack(
                [o_n_next[2], o_n_next[0], o_n_next[1], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[6], o_n_next[7],
                 o_n_next[8], o_n_next[9]]), False)

        agent4_memory.add(
            np.vstack([o_n[3], o_n[0], o_n[1], o_n[2], o_n[4], o_n[5], o_n[6], o_n[7], o_n[8], o_n[9]]),
            np.vstack([agent4_action, agent1_action, agent2_action, agent3_action, agent5_action, agent6_action,
                       agent7_action, agent8_action, agent9_action, agent10_action]),
            r_n[3], np.vstack(
                [o_n_next[3], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[4], o_n_next[5], o_n_next[6], o_n_next[7],
                 o_n_next[8], o_n_next[9]]), False)

        agent5_memory.add(
            np.vstack([o_n[4], o_n[0], o_n[1], o_n[2], o_n[3], o_n[5], o_n[6], o_n[7], o_n[8], o_n[9]]),
            np.vstack([agent5_action, agent1_action, agent2_action, agent3_action, agent4_action, agent6_action,
                       agent7_action, agent8_action, agent9_action, agent10_action]),
            r_n[4], np.vstack(
                [o_n_next[4], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[5], o_n_next[6], o_n_next[7],
                 o_n_next[8], o_n_next[9]]), False)

        agent6_memory.add(
            np.vstack([o_n[5], o_n[0], o_n[1], o_n[2], o_n[3], o_n[4], o_n[6], o_n[7], o_n[8], o_n[9]]),
            np.vstack([agent6_action, agent1_action, agent2_action, agent3_action, agent4_action, agent5_action,
                       agent7_action, agent8_action, agent9_action, agent10_action]),
            r_n[5], np.vstack(
                [o_n_next[5], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[6], o_n_next[7],
                 o_n_next[8], o_n_next[9]]), False)

        agent7_memory.add(
            np.vstack([o_n[6], o_n[0], o_n[1], o_n[2], o_n[3], o_n[4], o_n[5], o_n[6], o_n[8], o_n[9]]),
            np.vstack([agent7_action, agent1_action, agent2_action, agent3_action, agent4_action, agent5_action,
                       agent6_action, agent8_action, agent9_action, agent10_action]),
            r_n[6], np.vstack(
                [o_n_next[6], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[7],
                 o_n_next[8], o_n_next[9]]), False)

        agent8_memory.add(
            np.vstack([o_n[7], o_n[0], o_n[1], o_n[2], o_n[3], o_n[4], o_n[5], o_n[6], o_n[8], o_n[9]]),
            np.vstack([agent8_action, agent1_action, agent2_action, agent3_action, agent4_action, agent5_action,
                       agent6_action, agent7_action, agent9_action, agent10_action]),
            r_n[7], np.vstack(
                [o_n_next[7], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[6],
                 o_n_next[8], o_n_next[9]]), False)

        agent9_memory.add(
            np.vstack([o_n[8], o_n[0], o_n[1], o_n[2], o_n[3], o_n[4], o_n[5], o_n[6], o_n[7], o_n[9]]),
            np.vstack([agent9_action, agent1_action, agent2_action, agent3_action, agent4_action, agent5_action,
                       agent6_action, agent7_action, agent8_action, agent10_action]),
            r_n[8], np.vstack(
                [o_n_next[8], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[6],
                 o_n_next[7], o_n_next[9]]), False)

        agent10_memory.add(
            np.vstack([o_n[9], o_n[0], o_n[1], o_n[2], o_n[3], o_n[4], o_n[5], o_n[6], o_n[7], o_n[8]]),
            np.vstack([agent10_action, agent1_action, agent2_action, agent3_action, agent4_action, agent5_action,
                       agent6_action, agent7_action, agent8_action, agent9_action]),
            r_n[9], np.vstack(
                [o_n_next[9], o_n_next[0], o_n_next[1], o_n_next[2], o_n_next[3], o_n_next[4], o_n_next[5], o_n_next[6],
                 o_n_next[7], o_n_next[8]]), False)

        if i > 50000:
            # e *= 0.9999
            # agent1 train
            train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                        agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target, agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target, agent9_ddpg_target, agent10_ddpg_target])

            train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                        agent2_critic_target_update, sess, [agent1_ddpg_target, agent3_ddpg_target, agent4_ddpg_target, agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target, agent9_ddpg_target, agent10_ddpg_target])

            train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                        agent3_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target, agent4_ddpg_target, agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target, agent9_ddpg_target, agent10_ddpg_target])

            train_agent(agent4_ddpg, agent4_ddpg_target, agent4_memory, agent4_actor_target_update,
                        agent4_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent5_ddpg_target,
                         agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target, agent9_ddpg_target,
                         agent10_ddpg_target])

            train_agent(agent5_ddpg, agent5_ddpg_target, agent5_memory, agent5_actor_target_update,
                        agent5_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                         agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target, agent9_ddpg_target,
                         agent10_ddpg_target])

            train_agent(agent6_ddpg, agent6_ddpg_target, agent6_memory, agent6_actor_target_update,
                        agent6_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                         agent5_ddpg_target, agent7_ddpg_target, agent8_ddpg_target, agent9_ddpg_target,
                         agent10_ddpg_target])

            train_agent(agent7_ddpg, agent7_ddpg_target, agent7_memory, agent7_actor_target_update,
                        agent7_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                         agent5_ddpg_target, agent6_ddpg_target, agent8_ddpg_target, agent9_ddpg_target,
                         agent10_ddpg_target])

            train_agent(agent8_ddpg, agent8_ddpg_target, agent8_memory, agent8_actor_target_update,
                        agent8_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                         agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target, agent9_ddpg_target,
                         agent10_ddpg_target])

            train_agent(agent9_ddpg, agent9_ddpg_target, agent9_memory, agent9_actor_target_update,
                        agent9_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                         agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target,
                         agent10_ddpg_target])

            train_agent(agent10_ddpg, agent10_ddpg_target, agent10_memory, agent10_actor_target_update,
                        agent10_critic_target_update, sess,
                        [agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                         agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target,
                         agent9_ddpg_target])

        # 数据可视化
        for agent_index in range(10):
            summary_writer.add_summary(
                sess.run(agent_reward_op[agent_index], {agent_reward_v[agent_index]: r_n[agent_index]}), i)
            summary_writer.add_summary(sess.run(agent_a1_op[agent_index], {agent_a1[agent_index]: a[agent_index][1]}),
                                       i)
            summary_writer.add_summary(sess.run(agent_a2_op[agent_index], {agent_a2[agent_index]: a[agent_index][3]}),
                                       i)
            summary_writer.add_summary(
                sess.run(reward_100_op[agent_index], {reward_100[agent_index]: np.mean(reward_100_list[agent_index])}),
                i)

        o_n = o_n_next

        if i % 1000 == 0:
            saver.save(sess, './three_ma_weight/' + str(i) + '.cptk')

    sess.close()
