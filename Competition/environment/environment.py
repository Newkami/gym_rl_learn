# 定义步兵类
from data_generation import Task_Generator
from gym import spaces
from target import Army
import random
import secrets


class Env():
    def __init__(self, num_infantry, num_catapult, num_outpost, num_shieldarray, num_buildings, num_Ballista,
                 num_agent):
        # try:
        #     self.target_check(num_infantry, num_catapult, num_outpost, num_shieldarray, num_buildings)
        # except Exception:
        #     print("任务目标数量错误，暂不生成目标")

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

    # def target_check(self, num_infantry, num_catapult, num_outpost, num_shieldarray, num_buildings):
    #     if not 0 < num_infantry <= 10:
    #         print("兵马类目标数量设置不合理")
    #         return False
    #     if not 0 < num_catapult <= 20:
    #         print("投石车目标数量设置不合理")
    #         return False
    #     if not 0 < num_outpost <= 20:
    #         print("前哨站目标数量设置不合理")
    #         return False
    #     if not 0 < num_shieldarray <= 20:
    #         print("盾牌阵目标数量设置不合理")
    #         return False
    #     if not 0 < num_buildings <= 30:
    #         print("建筑类目标数量设置不合理")
    #         return False
    #     if not num_infantry+num_catapult+num_outpost+num_shieldarray+num_buildings==60:
    #         print("总目标数不为60，请重新设定")
    #         return False
    #     return True

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
        # 使用传入的动作减少相应目标的血量

        # 2.建立动作到目标的映射
        # 3.减少HP值

        # 反制阶段
        # 魏国军队根据剩余军队数量进行反制打击
        # 减少o_n中弩车数量

        # 计算Reward

        # 判断是否结束

        # 返回o_n',r_n,d_n
        # 每个军队智能体的动作选择范围应该为[0-14]
        # 0-4 代表使用弩车攻击兵马类、投石车。。。。
        # 5-9 代表使用火弩攻击兵马类、投石车。。。
        # 10-14 代表使用药弩攻击兵马类、投石车。。。
        # action_list 应该为10个军队智能体的动作 [ 1,3,5...13] 共30个数字
        reward_list = []  # 奖励值列表
        for army_index in range(len(action_list)):
            # 1.判断动作是否有效（目标是否在攻击范围内 以及目标是否还存活）
            action_avalible_list = []  # 可用动作列表
            target_list = []  # 可打击目标列表
            reward = 0
            for action_index in range(len(action_list[army_index])):
                if 0 <= action_list[army_index][action_index] <= 4:
                    crosstype = 0
                elif 5 <= action_list[army_index][action_index] <= 9:
                    crosstype = 1
                elif 10 <= action_list[army_index][action_index] <= 14:
                    crosstype = 2
                target_type = action_list[army_index][action_index] % 5  # 目标建筑类型
                target = self.targetChoice(target_type)
                x, y, z = self.isActionAvailable(self.army_list[army_index], crosstype,
                                                 target)
                if x and y and z:
                    print("随机挑选的目标为:" + target.name + " 目标当前HP:", target.HP)
                    action_avalible_list.append(action_list[army_index][action_index])
                    target_list.append(target)
                # 该根据不同的有效目标设计不同的惩罚项
                # (1) 若弓箭类型足够，距离也够，但打击目标已经被摧毁了 此时是否要给予惩罚，弩箭的数量是否需要减少？
                elif x and y and not z:
                    reward -= 10
                # (2) 若弓箭类型不足够，距离够
                # (3) 若弓箭类型足够，距离不够
                else:
                    reward -= 15
                    print("该次动作选择不可用")
            # 2.实际打击阶段
            if self.army_list[army_index].num_ballista == 0:
                continue
            if self.army_list[army_index].num_ballista >= len(
                    action_avalible_list):  # 若剩余弩车数量大于等于可执行动作数量, 执行可用动作列表中的全部动作
                for i in range(len(target_list)):
                    # 消耗弩箭

                    crosstype = self.getCrossbowType(action_avalible_list[i])

                    if crosstype == 0:
                        self.army_list[army_index].num_bolt -= 1
                    elif crosstype == 1:
                        self.army_list[army_index].num_firebolt -= 1
                    elif crosstype == 2:
                        self.army_list[army_index].num_med_bolt -= 1
                    # 目标被实际打击
                    damage = target_list[i].BeAttacked(crosstype, self.shieldarray_list)

                    print("打击后的目标" + target_list[i].name + "血量为：", target_list[i].HP, "造成伤害", damage)
                    if target_list[i].target_type == 0:
                        reward += (damage + target_list[i].strike_ability) * 1.5
                    elif target_list[i].target_type == 1:
                        reward += (damage + target_list[i].GetCurrentAbility(target_list[i].HP)) * 1.5
                    else:
                        reward += damage * 1.5
                    # print(target.target_type, reward, damage)
                    if target_list[i].HP == 0:  # 若目标被击毁 增加奖励值
                        reward += 50
                reward_list.append(reward)

            else:  # 弩车剩余数量不足，则随机执行等于弩车数量的动作
                temp_index = secrets.SystemRandom().sample(list(range(len(action_avalible_list))),
                                                           self.army_list[army_index].num_ballista)  # 随机动作值索引列表
                for i in temp_index:
                    # 消耗弩箭
                    crosstype = self.getCrossbowType(action_avalible_list[i])

                    if crosstype == 0:
                        self.army_list[army_index].num_bolt -= 1
                    elif crosstype == 1:
                        self.army_list[army_index].num_firebolt -= 1
                    elif crosstype == 2:
                        self.army_list[army_index].num_med_bolt -= 1

                    # 目标被实际打击
                    damage = target_list[i].BeAttacked(crosstype, self.shieldarray_list)
                    print("打击后的目标" + target_list[i].name + "血量为：", target_list[i].HP, "造成伤害", damage)
                    if target_list[i].target_type == 0:
                        reward += (damage + target_list[i].strike_ability) * 1.5
                    elif target_list[i].target_type == 1:
                        reward += (damage + target_list[i].GetCurrentAbility(target_list[i].HP)) * 1.5
                    else:
                        reward += damage * 1.5
                    if target_list[i].HP == 0:
                        reward += 50
                    # print(target.target_type, reward, damage)

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

    def targetChoice(self, target_type):
        target = None
        if target_type == 0:
            target = random.choice(self.infantry_list)
        elif target_type == 1:
            target = random.choice(self.catapult_list)
        elif target_type == 2:
            target = random.choice(self.outpost_list)
        elif target_type == 3:
            target = random.choice(self.shieldarray_list)
        elif target_type == 4:
            target = random.choice(self.buildingobjective_list)

        return target

    def isActionAvailable(self, army, crosstype, target):
        crossbow_b = False
        distance_b = False
        alive_b = True
        # 判断动作是否可用
        # 1.判断当前军队的弩箭数量是否足够
        if self.isCrossbowEnough(army, crosstype):  # 若当前军队的特定弩箭类型数目足够
            crossbow_b = True
            # 2.判断当前打击目标是否足够距离
            distance_b = self.isTargetInDistance(army, target, crosstype)
            # 3.判断当前打击目标是否存活
            alive_b = target.HP > 0
        return crossbow_b, distance_b, alive_b

    def isCrossbowEnough(self, army, crossbow_type):
        # 根据弩箭类型判断数量是否足够
        if crossbow_type == 0:
            return army.num_bolt > 0
        elif crossbow_type == 1:
            return army.num_firebolt > 0
        elif crossbow_type == 2:
            return army.num_med_bolt > 0

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

    def getCrossbowType(self, index):
        if 0 <= index <= 4:
            crosstype = 0
        elif 5 <= index <= 9:
            crosstype = 1
        elif 10 <= index <= 14:
            crosstype = 2
        return crosstype

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
# env = Env(10, 10, 10, 10, 20, 10, 10)
# a = env.reset()
#
# action_list = [
#     [
#         1, 5, 9, 10, 11, 2
#     ],
#     [
#         3, 6, 1, 0, 12, 14
#     ]
# ]
# print(env.getCurrentState())
# print("----------------------------------------")
# env.step(action_list)
# print(env.getCurrentState())
# print("----------------------------------------")
# env.step(action_list)
# print(env.getCurrentState())
# print("----------------------------------------")
# env.step(action_list)
# print(env.getCurrentState())
# print("----------------------------------------")
# env.step(action_list)
# print(env.getCurrentState())
# print("----------------------------------------")
# env.step(action_list)
# print(env.getCurrentState())
# print("----------------------------------------")
# print(env.step(action_list))
