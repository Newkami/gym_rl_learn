# infantry 步兵
# Ballista 弩车
# Catapult 投石车
import torch
class Infantry():
    def __init__(self, name, HP, pos_x, pos_y, target_type, strike_ability, strike_army):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.strike_ability = strike_ability
        self.target_type = target_type
        self.HP = HP
        self.MAX_HP = HP
        self.strike_army = strike_army

    def HP_Recovery(self):
        if self.HP != 0:
            self.HP += 0.2 * self.MAX_HP  # 恢复最大生命值的百分之20
            if self.HP <= self.MAX_HP:
                return
            self.HP = self.MAX_HP

    # def Destroy_Ballista(self, army):  # army为指定的军队
    #     army.num_ballista -= self.strike_ability
    #     return

    def BeAttacked(self, crossbow_type, shield_array_list):
        """
        :param crossbow_type: 弓弩的类型
        :param shield_array_list: 盾牌阵列表
        :return: 返回值为是否成功击中并造成伤害
        :type shield_array_list: list<ShieldArray>
        """
        dc_list = []
        damage = 0
        for i in range(len(shield_array_list)):
            if shield_array_list[i].IsInSheild(self.pos_x, self.pos_y):  # 判断是否在某一盾牌阵内 并获取相应的防护系数
                dc_list.append(shield_array_list[i].GetCurrentDefendCofficient)
        if len(dc_list) != 0:
            defend_coefficient = max(dc_list)
        else:
            defend_coefficient = 0

        if crossbow_type == 0:
            damage = 40
        elif crossbow_type == 1:
            damage = 50
        elif crossbow_type == 2:
            damage = 60

        if self.HP >= damage * (1 - defend_coefficient):
            self.HP -= damage * (1 - defend_coefficient)
            actual_damage = damage * (1 - defend_coefficient)
        else:
            actual_damage = self.HP
            self.HP = 0
        return actual_damage


class Catapult:
    def __init__(self, name, HP, pos_x, pos_y, target_type, strike_army):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.HP = HP
        self.target_type = target_type
        self.strike_army = strike_army

    def GetCurrentAbility(self, HP):
        counteract_nums = 5
        if HP > 450:
            counteract_nums = 5
        elif 300 < HP <= 450:
            counteract_nums = 4
        elif 200 < HP <= 300:
            counteract_nums = 3
        elif 100 < HP <= 200:
            counteract_nums = 2
        elif 30 < HP <= 100:
            counteract_nums = 1
        elif 0 <= HP <= 30:
            counteract_nums = 0
        return counteract_nums

    def BeAttacked(self, crossbow_type, shield_array_list):
        """
        :param crossbow_type: 弓弩的类型
        :param shield_array_list: 盾牌阵列表
        :return: 返回值为是否成功击中并造成伤害
        :type shield_array_list: list<ShieldArray>
        """
        dc_list = []
        damage = 0
        for i in range(len(shield_array_list)):
            if shield_array_list[i].IsInSheild(self.pos_x, self.pos_y):  # 判断是否在某一盾牌阵内 并获取相应的防护系数
                dc_list.append(shield_array_list[i].GetCurrentDefendCofficient)
        if len(dc_list) != 0:
            defend_coefficient = max(dc_list)
        else:
            defend_coefficient = 0

        if crossbow_type == 0:
            damage = 20
        elif crossbow_type == 1:
            damage = 30
        elif crossbow_type == 2:
            damage = 40

        if self.HP >= damage * (1 - defend_coefficient):
            self.HP -= damage * (1 - defend_coefficient)
            actual_damage = damage * (1 - defend_coefficient)
        else:
            actual_damage = self.HP
            self.HP = 0
        return actual_damage


class Outpost:  # 定义前哨站
    def __init__(self, name, HP, pos_x, pos_y, target_type, detection_radius):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.HP = HP
        self.target_type = target_type
        self.detection_radius = detection_radius
        self.initial_HP = HP  # 初始HP
        self.initial_dr = detection_radius  # 初始监测半径

    def IsInDetected(self, tar_x, tar_y):  # 判断目标是否处于探测半径内
        return (tar_x - self.pos_x) ** 2 + (tar_y - self.pos_y) ** 2 <= self.detection_radius ** 2

    def GetHP_DesenctDegree(self, current_HP):
        descent_degree = (self.initial_HP - current_HP) / self.initial_HP
        return descent_degree

    def GetCurrent_Detection_Radius(self):
        descent_degree = self.GetHP_DesenctDegree(self.HP)
        if descent_degree == 0:
            self.detection_radius = self.initial_dr
        elif 0 < descent_degree <= 0.2:
            self.detection_radius = 0.8 * self.initial_dr
        elif 0.2 < descent_degree <= 0.5:
            self.detection_radius = 0.5 * self.initial_dr
        elif 0.5 < descent_degree <= 1:
            self.detection_radius = 0
        return self.detection_radius

    def BeAttacked(self, crossbow_type, shield_array_list):
        """
        :param crossbow_type: 弓弩的类型  0 弓弩 1 火弩 2 药弩
        :param shield_array_list: 盾牌阵列表
        :return: 返回值为是否成功击中并造成伤害
        :type shield_array_list: list<ShieldArray>
        """
        dc_list = []
        damage = 0
        for i in range(len(shield_array_list)):
            if shield_array_list[i].IsInSheild(self.pos_x, self.pos_y):  # 判断是否在某一盾牌阵内 并获取相应的防护系数
                dc_list.append(shield_array_list[i].GetCurrentDefendCofficient)
        if len(dc_list) != 0:
            defend_coefficient = max(dc_list)
        else:
            defend_coefficient = 0

        if crossbow_type == 0:
            damage = 30
        elif crossbow_type == 1:
            damage = 40
        elif crossbow_type == 2:
            damage = 60

        if self.HP >= damage * (1 - defend_coefficient):
            self.HP -= damage * (1 - defend_coefficient)
            actual_damage = damage * (1 - defend_coefficient)
        else:
            actual_damage = self.HP
            self.HP = 0
        return actual_damage


class ShieldArray:  # 定义盾牌阵
    def __init__(self, name, HP, pos_x, pos_y, target_type, defend_radius, defend_coefficient):
        self.outpost_list = None
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.HP = HP
        self.target_type = target_type
        self.defend_coefficient = defend_coefficient
        self.initial_HP = HP
        self.initial_dc = defend_coefficient
        self.defend_radius = defend_radius  # 定义防卫半径

    def set_outpostlist(self, outpost_list):
        self.outpost_list = outpost_list

    def IsInSheild(self, tar_x, tar_y):  # 判断目标是否处于盾牌阵内
        return (tar_x - self.pos_x) ** 2 + (tar_y - self.pos_y) ** 2 <= self.defend_radius ** 2

    def GetHP_DesenctDegree(self, current_HP):  # 获取相对初始生命值下降程度
        descent_degree = (self.initial_HP - current_HP) / self.initial_HP
        return descent_degree

    def GetOutpostDetection_r(self, outpost_list):  # 未完成！！！
        """
        :param outpost_list: 前哨站列表
        :type outpost_list:list<Outpost>
        :return: 获取最大前哨站探测半径
        """

        dr_list = []
        for i in range(len(outpost_list)):
            if outpost_list[i].IsInDetected(self.pos_x, self.pos_y):
                dr_list.append(outpost_list[i].GetCurrent_Detection_Radius())  # 添加所有覆盖盾牌阵的前哨站的探测半径

        if len(dr_list) != 0:
            detection_radius = max(dr_list)
        else:
            detection_radius = 0
        return detection_radius

    @property
    def GetCurrentDefendCofficient(self):  # 获取当前防护系数
        detection_r = self.GetOutpostDetection_r(self.outpost_list)
        descent_degree = self.GetHP_DesenctDegree(self.HP)
        dc_descent_degree = 0
        if detection_r == 0:
            dc_descent_degree = 1
        if 0 <= detection_r <= 100:
            if 0 <= descent_degree <= 0.1:
                dc_descent_degree = 0.5
            elif 0.1 < descent_degree <= 0.3:
                dc_descent_degree = 0.6
            elif 0.3 < descent_degree <= 0.7:
                dc_descent_degree = 0.8
            elif 0.7 < descent_degree <= 1:
                dc_descent_degree = 1
        if 100 < detection_r <= 200:
            if 0 <= descent_degree <= 0.1:
                dc_descent_degree = 0.3
            elif 0.1 < descent_degree <= 0.3:
                dc_descent_degree = 0.4
            elif 0.3 < descent_degree <= 0.7:
                dc_descent_degree = 0.7
            elif 0.7 < descent_degree <= 1:
                dc_descent_degree = 1
        if 200 < detection_r <= 300:
            if 0 <= descent_degree <= 0.1:
                dc_descent_degree = 0.1
            elif 0.1 < descent_degree <= 0.3:
                dc_descent_degree = 0.3
            elif 0.3 < descent_degree <= 0.7:
                dc_descent_degree = 0.5
            elif 0.7 < descent_degree <= 1:
                dc_descent_degree = 1
        if detection_r > 300:
            if 0 <= descent_degree <= 0.1:
                dc_descent_degree = 0.1
            elif 0.1 < descent_degree <= 0.3:
                dc_descent_degree = 0.2
            elif 0.3 < descent_degree <= 0.7:
                dc_descent_degree = 0.4
            elif 0.7 < descent_degree <= 1:
                dc_descent_degree = 1

        self.defend_coefficient = (1 - dc_descent_degree) * self.defend_coefficient
        return self.defend_coefficient

    def BeAttacked(self, crossbow_type, shield_array_list):
        '''
        :param crossbow_type: 弓弩的类型
        :param shield_array_list: 盾牌阵列表
        :return: 返回值为是否成功击中并造成伤害
        :type shield_array_list: list<ShieldArray>
        '''
        dc_list = []
        damage = 0
        for i in range(len(shield_array_list)):
            if shield_array_list[i].IsInSheild(self.pos_x, self.pos_y):  # 判断是否在某一盾牌阵内 并获取相应的防护系数
                dc_list.append(shield_array_list[i].GetCurrentDefendCofficient)
        if len(dc_list) != 0:
            defend_coefficient = max(dc_list)
        else:
            defend_coefficient = 0

        if crossbow_type == 0:
            damage = 20
        elif crossbow_type == 1:
            damage = 30
        elif crossbow_type == 2:
            damage = 50

        if self.HP >= damage * (1 - defend_coefficient):
            self.HP -= damage * (1 - defend_coefficient)
            actual_damage = damage * (1 - defend_coefficient)
        else:
            actual_damage = self.HP
            self.HP = 0
        return actual_damage

class BuildingObjective:  # 定义建筑目标
    def __init__(self, name, HP, pos_x, pos_y, target_type):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.HP = HP
        self.target_type = target_type

    def BeAttacked(self, crossbow_type, shield_array_list):
        """
        :param crossbow_type: 弓弩的类型
        :param shield_array_list: 盾牌阵列表
        :return: 返回值为是否成功击中并造成伤害
        :type shield_array_list: list<ShieldArray>
        """
        dc_list = []
        damage = 0
        for i in range(len(shield_array_list)):
            if shield_array_list[i].IsInSheild(self.pos_x, self.pos_y):  # 判断是否在某一盾牌阵内 并获取相应的防护系数
                dc_list.append(shield_array_list[i].GetCurrentDefendCofficient)

        if len(dc_list) != 0:
            defend_coefficient = max(dc_list)
        else:
            defend_coefficient = 0

        if crossbow_type == 0:
            damage = 0
        elif crossbow_type == 1:
            damage = 20
        elif crossbow_type == 2:
            damage = 60

        if self.HP >= damage * (1 - defend_coefficient):
            self.HP -= damage * (1 - defend_coefficient)
            actual_damage = damage * (1 - defend_coefficient)
        else:
            actual_damage = self.HP
            self.HP = 0
        return actual_damage


class Army:
    def __init__(self, name, num_ballista, pos_x, pos_y, num_bolt, num_firebolt, num_med_bolt):
        """
        :param name: 军队名称
        :param num_ballista: 弩车数量
        :param pos_x: 军队位置x
        :param pos_y: 军队位置y
        :param num_bolt: 普通弩数量
        :param num_firebolt: 火弩数量
        :param num_medbolt: 药弩数量
        """
        self.name = name
        self.num_ballista = num_ballista
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.num_bolt = num_bolt
        self.num_firebolt = num_firebolt
        self.num_med_bolt = num_med_bolt

    def BeAttacked(self, num_U):
        if self.num_ballista - num_U >= 0:
            self.num_ballista -= num_U
        else:
            self.num_ballista = 0


