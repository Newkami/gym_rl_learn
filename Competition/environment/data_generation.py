import pandas as pd
from target import Infantry, Catapult, Outpost, ShieldArray, BuildingObjective


class Task_Generator:
    def __init__(self, num_infantry, num_catapult, num_outpost, num_shield_array, num_buildings):
        self.num_infantry = num_infantry
        self.num_catapult = num_catapult
        self.num_outpost = num_outpost
        self.num_shield_array = num_shield_array
        self.num_buildings = num_buildings
        self.infantry_df = pd.read_excel("../data/Infantry.xlsx")
        self.catapult_df = pd.read_excel('../data/Catapult.xlsx')
        self.outpost_df = pd.read_excel('../data/Outpost.xlsx')
        self.shield_df = pd.read_excel('../data/ShieldArray.xlsx')
        self.buildings_df = pd.read_excel('../data/Buildings.xlsx')

    def target_generation(self):
        infantry_list = []
        catapult_list = []
        outpost_list = []
        shield_array_list = []
        buildings_list = []

        self.infantry_df = self.infantry_df.sample(self.num_infantry)
        self.catapult_df = self.catapult_df.sample(self.num_catapult)
        self.outpost_df = self.outpost_df.sample(self.num_outpost)
        self.shield_df = self.shield_df.sample(self.num_shield_array)
        self.buildings_df = self.buildings_df.sample(self.num_buildings)

        for index, row in self.infantry_df.iterrows():
            infantry_list.append(
                Infantry(row['name'], row['HP'], row['pos_x'], row['pos_y'], row['target_type'], row['strike_ability'],
                         row['strike_army'], row['idx_value']))

        for index, row in self.catapult_df.iterrows():
            catapult_list.append(
                Catapult(row['name'], row['HP'], row['pos_x'], row['pos_y'], row['target_type'], row['strike_army'], row['idx_value']))

        for index, row in self.outpost_df.iterrows():
            outpost_list.append(
                Outpost(row['name'], row['HP'], row['pos_x'], row['pos_y'], row['target_type'],
                        row['detection_radius'], row['idx_value']))

        for index, row in self.shield_df.iterrows():
            shield_array_list.append(
                ShieldArray(row['name'], row['HP'], row['pos_x'], row['pos_y'], row['target_type'],
                            row['defend_radius'], row['defend_coefficient'], row['idx_value']))

        for index, row in self.buildings_df.iterrows():
            buildings_list.append(
                BuildingObjective(row['name'], row['HP'], row['pos_x'], row['pos_y'], row['target_type'], row['idx_value']))

        for i in range(len(shield_array_list)):
            shield_array_list[i].set_outpostlist(outpost_list)

        return infantry_list, catapult_list, outpost_list, shield_array_list, buildings_list
