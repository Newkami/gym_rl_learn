# 定义步兵类
import torch

from data_generation import Task_Generator
from gym import spaces
from target import Army
import random
import secrets
import numpy as np


class Env():
    def __init__(self, num_infantry, num_catapult, num_outpost, num_shieldarray, num_buildings, num_Ballista,
                 num_agent):

        self.num_shieldArray = num_shieldarray
        self.num_outpost = num_outpost
        self.num_catapult = num_catapult
        self.num_infantry = num_infantry
        self.num_buildings = num_buildings
        self.num_Ballista = num_Ballista
        self.num_agent = num_agent
        self.action_space = spaces.Discrete(180)
        self.observation_space = None

        self.infantry_list = []
        self.catapult_list = []
        self.outpost_list = []
        self.shieldarray_list = []
        self.buildingobjective_list = []
        self.army_list = []

    def InitializeArmy(self):
        self.army_list.append(Army("A", 30, 70, 60, 100, 100, 100))
        self.army_list.append(Army("B", 30, 60, 180, 200, 100, 0))
        self.army_list.append(Army("C", 30, 40, 360, 0, 150, 150))
        self.army_list.append(Army("D", 30, 10, 680, 100, 0, 200))
        self.army_list.append(Army("E", 30, 20, 370, 0, 0, 300))
        self.army_list.append(Army("F", 30, 140, 750, 300, 0, 0))
        self.army_list.append(Army("G", 30, 150, 260, 300, 0, 0))
        self.army_list.append(Army("H", 30, 90, 620, 50, 150, 100))
        self.army_list.append(Army("I", 30, 80, 350, 0, 180, 120))
        self.army_list.append(Army("J", 30, 110, 390, 300, 0, 0))

    def InitializeEnv(self):
        task_generator = Task_Generator(self.num_infantry, self.num_catapult, self.num_outpost,
                                        self.num_shieldArray,
                                        self.num_buildings)
        self.infantry_list, self.catapult_list, self.outpost_list, self.shieldarray_list, self.buildingobjective_list = task_generator.target_generation()

    def reset(self):
        self.infantry_list.clear()
        self.catapult_list.clear()
        self.shieldarray_list.clear()
        self.outpost_list.clear()
        self.buildingobjective_list.clear()
        self.InitializeEnv()
        self.InitializeArmy()
        o_n = []
        for i in range(len(self.army_list)):
            o_n.append(self.getCurrentObs(i))
        return o_n

    def step(self, action_list):
        # 每个军队弩车的动作选择范围应该为[1,1,1,1,0,0,0......0]
        # 索引值
        # 0 代表no-op
        # 1-100 代表使用弩车攻击兵马类、投石车。。。。
        # 101-200 代表使用火弩攻击兵马类、投石车。。。
        # 201-300 代表使用药弩攻击兵马类、投石车。。。
        # action_list 应该为10个军队智能体的动作 每个军队的动作为[1,3,5...13] 共30个数字
        reward_list = []  # 奖励值列表
        for army_index, actions in enumerate(action_list):
            reward = 0
            for action_index, action in enumerate(actions):
                if action == 0:
                    continue  # no-op 不作操作
                crosstype = self._get_crosstype(action)  # 获取箭弩类型
                target = self._get_target(action)  # 获取被击打的目标
                if not self.isTargetInDistance(self.army_list[army_index], target, crosstype):  # 如果超出距离限制，给予负奖励
                    print("该目标不在打击范围内，奖励值降低")
                    print("---------------------------------------------------------------")
                    reward -= 100
                    continue  # 跳过该动作
                print("打击前的目标" + target.name + "血量为：", target.HP)
                damage = target.BeAttacked(crosstype, self.shieldarray_list)
                print("打击后的目标" + target.name + "血量为：", target.HP, "造成伤害", damage)
                print("---------------------------------------------------------------")
                if target.target_type == 0:
                    reward += (damage + target.strike_ability) * 1.5
                elif target.target_type == 1:
                    reward += (damage + target.GetCurrentAbility(target.HP)) * 1.5
                else:
                    reward += damage * 1.5
                if target.HP == 0:
                    reward += 50

            reward_list.append(reward)
        # 3.反制阶段
        # 3.1 兵马类目标反制

        for infantry in self.infantry_list:
            if infantry.HP > 0:  # 若兵马类目标血量不为0
                i = ord(infantry.strike_army) - 65
                self.army_list[i].BeAttacked(infantry.strike_ability)
                infantry.HP_Recovery()  # 反制后恢复血量

        # 3.2 投石车类目标反制
        for catapult in self.catapult_list:
            if catapult.HP > 0:
                i = ord(catapult.strike_army) - 65
                self.army_list[i].BeAttacked(catapult.GetCurrentAbility(catapult.HP))

        print(reward_list)
        # 反制阶段结束

        done = self.isDone()

        o_n = []
        for i in range(len(self.army_list)):
            o_n.append(self.getCurrentObs(i))
        return o_n, reward_list, done

    def _get_crosstype(self, action_value):
        if 0 < action_value <= 100:
            cross_type = 0
        elif 100 < action_value <= 200:
            cross_type = 1
        elif 200 < action_value <= 300:
            cross_type = 2

        return cross_type

    def _get_target(self, action_value):
        target_idx = action_value
        if 100 < action_value <= 200:
            target_idx -= 100
        elif 200 < action_value <= 300:
            target_idx -= 200

        target = None
        if 1 <= target_idx <= 10:
            for infantry in self.infantry_list:
                if infantry.idx_value == target_idx:
                    target = infantry
                    break
        if 11 <= target_idx <= 30:
            for catapult in self.catapult_list:
                if catapult.idx_value == target_idx:
                    target = catapult
                    break
        if 31 <= target_idx <= 50:
            for outpost in self.outpost_list:
                if outpost.idx_value == target_idx:
                    target = outpost
                    break
        if 51 <= target_idx <= 70:
            for shield in self.shieldarray_list:
                if shield.idx_value == target_idx:
                    target = shield
                    break
        if 71 <= target_idx <= 100:
            for building in self.buildingobjective_list:
                if building.idx_value == target_idx:
                    target = building
                    break

        if target == None:
            print("未找到该目标")
        else:
            return target

    def isDone(self):
        """
                # 判断当前回合是否结束
                # 1. 军队全部阵亡
                # 2. 目标全部被击毁
        :return:
        """
        for army in self.army_list:
            if army.num_ballista != 0:
                return False

        for infantry in self.infantry_list:
            if infantry.HP != 0:
                return False
        for catapult in self.catapult_list:
            if catapult.HP != 0:
                return False
        for outpost in self.outpost_list:
            if outpost.HP != 0:
                return False
        for shieldarray in self.shieldarray_list:
            if shieldarray.HP != 0:
                return False
        for building in self.buildingobjective_list:
            if building.HP != 0:
                return False
        return True

    def isTargetInDistance(self, army, target, cross_type):
        """
        :param army: 当前军队
        :param target:
        :param cross_type:
        :return:
        """
        if cross_type == 0:
            return (target.pos_x - army.pos_x) ** 2 + (target.pos_y - army.pos_y) ** 2 <= 800 ** 2
        elif cross_type == 1:
            return (target.pos_x - army.pos_x) ** 2 + (target.pos_y - army.pos_y) ** 2 <= 1000 ** 2
        elif cross_type == 2:
            return (target.pos_x - army.pos_x) ** 2 + (target.pos_y - army.pos_y) ** 2 <= 600 ** 2

    def getCurrentState(self):
        s = []
        for i in range(len(self.army_list)):
            s.append(self.army_list[i].num_ballista)
        return s

    def getCurrentObs(self, i):
        o = []
        o.append(self.army_list[i].num_ballista)
        o.append(self.army_list[i].num_bolt)
        o.append(self.army_list[i].num_firebolt)
        o.append(self.army_list[i].num_med_bolt)
        return o

    def get_army_by_name(self, name):
        return self.army_list[ord(name) - 65]

    def get_avail_army_actions(self, army_name):
        # 返回的是单个弩车的可用动作
        army = self.get_army_by_name(army_name)
        avail_actions = [0] * 301
        avail_actions[0] = 1  # 应该允许no-op的存在 若弩车被击毁视为no-op

        bolt_b = True if army.num_bolt > 0 else False

        firebolt_b = True if army.num_firebolt > 0 else False

        medbolt_b = True if army.num_med_bolt > 0 else False

        for infantry in self.infantry_list:
            if bolt_b:
                avail_actions[infantry.idx_value] = 1
            if firebolt_b:
                avail_actions[infantry.idx_value + 1 * 100] = 1
            if medbolt_b:
                avail_actions[infantry.idx_value + 2 * 100] = 1
        for catapult in self.catapult_list:
            if bolt_b:
                avail_actions[catapult.idx_value] = 1
            if firebolt_b:
                avail_actions[catapult.idx_value + 1 * 100] = 1
            if medbolt_b:
                avail_actions[catapult.idx_value + 2 * 100] = 1
        for outpost in self.outpost_list:
            if bolt_b:
                avail_actions[outpost.idx_value] = 1

            if firebolt_b:
                avail_actions[outpost.idx_value + 1 * 100] = 1
            if medbolt_b:
                avail_actions[outpost.idx_value + 2 * 100] = 1
        for shieldarray in self.shieldarray_list:
            if bolt_b:
                avail_actions[shieldarray.idx_value] = 1

            if firebolt_b:
                avail_actions[shieldarray.idx_value + 1 * 100] = 1

            if medbolt_b:
                avail_actions[shieldarray.idx_value + 2 * 100] = 1
        for building in self.buildingobjective_list:
            if bolt_b:
                avail_actions[building.idx_value] = 1
            if firebolt_b:
                avail_actions[building.idx_value + 1 * 100] = 1
            if medbolt_b:
                avail_actions[building.idx_value + 2 * 100] = 1

        return avail_actions

    def get_avail_actions(self):
        avail_actions = []
        for army in self.army_list:
            avail_army = self.get_avail_army_actions(army.name)
            avail_actions.append(avail_army)

        return avail_actions


env = Env(10, 10, 10, 10, 20, 30, 10)
env.reset()
action = env.get_avail_army_actions('A')
action = np.nonzero(action)[0]
action = action.tolist()
action = [action]
print(action)
env.step(action)
