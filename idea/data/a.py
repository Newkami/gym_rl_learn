import re

std_dic = {}


def save_std():
    rule = '[\u4e00-\u9fa5]'
    select_regex_hand = re.compile(rule)
    select_regex_tail = re.compile('---(\d*?)---(\d*?)---(\d*?)---(\d*)')
    with open('学生信息数据库.txt', 'r', encoding='utf-8') as f:
        list_name = f.readlines()
        for m in range(len(list_name)):
            hand = select_regex_hand.findall(list_name[m])
            hand = ''.join(hand)
            std_dic[hand] = []
            for k in range(4):
                std_dic[hand].append(select_regex_tail.findall(list_name[m])[0][k])


def std_msg():
    name = input('输入你的名字：')
    score = input('输入你的英语、数学、专业课成绩，中间用空格隔开：')
    regex = re.compile(name)
    with open('学生信息数据库.txt', 'r', encoding='utf-8') as f:
        list_name = f.readlines()
        for n in range(len(list_name)):
            if regex.findall(list_name[n]):
                return '数据库中已有此人。'
        std_dic[name] = list(map(int, score.split(' ')))
        sum_score = std_dic[name][0] + std_dic[name][1] + std_dic[name][2]
        std_dic[name].append(sum_score)
        insert_msg(name)
        return '成功写入记录'


def insert_msg(name):
    with open('/Users/wuyufeng/Desktop/学生信息数据库.txt', 'a', encoding='utf-8') as f:
        f.write(name + '---' + str(std_dic[name][0]) + '---' + str(std_dic[name][1]) + '---' + str(
                std_dic[name][2]) + '---' + str(std_dic[name][3]))
        f.write('\n')


def show_msg():
    name = input('输入你要查找的名字：')
    regex = re.compile('[\u4e00-\u9fa5]')
    with open('学生信息数据库.txt', 'r', encoding='utf-8') as f:
        list_name = f.readlines()
        for i in range(len(list_name)):
            f_name = regex.findall(list_name[i])
            f_name = ''.join(f_name)
            if not regex.findall(list_name[i]):
                continue
            elif f_name == name:
                return list_name[i]
    return '未查找到此人'


def show_all_msg():
    with open('学生信息数据库.txt', 'r', encoding='utf-8') as f:
        all_record = f.readlines()
        for i in range(len(all_record)):
            record = all_record[i].replace('\n','')
            print(record)
        return ''


def del_record():
    name = input('输入你要删除的同学的名字：')
    rule = '[\u4e00-\u9fa5]'
    select_regex_hand = re.compile(rule)
    select_regex_tail = re.compile('---(\d*?)---(\d*?)---(\d*?)---(\d*)')
    with open('学生信息数据库.txt', 'r', encoding='utf-8') as f:
        list_name = f.readlines()
        for m in range(len(list_name)):
            hand = select_regex_hand.findall(list_name[m])
            hand = ''.join(hand)
            std_dic[hand] = []
            for k in range(4):
                std_dic[hand].append(select_regex_tail.findall(list_name[m])[0][k])
        if not std_dic.get(name):
            return '未查找到此人'
        else:
            del std_dic[name]
    with open('学生信息数据库.txt', 'w', encoding='utf-8') as f:
            for j in std_dic.keys():
                f.write(j + '---' + str(std_dic[j][0]) + '---' + str(std_dic[j][1]) + '---' + str(std_dic[j][2]) +
                        '---' + str(std_dic[j][3]))
                f.write('\n')
            return '成功删除'


def alter_record():
    name = input('输入你要修改的同学的名字：')
    rule = '[\u4e00-\u9fa5]'
    select_regex_hand = re.compile(rule)
    select_regex_tail = re.compile('---(\d*?)---(\d*?)---(\d*?)---(\d*)')
    with open('学生信息数据库.txt', 'r', encoding='utf-8') as f:
        list_name = f.readlines()
        for m in range(len(list_name)):
            hand = select_regex_hand.findall(list_name[m])
            hand = ''.join(hand)
            std_dic[hand] = []
            for k in range(4):
                std_dic[hand].append(select_regex_tail.findall(list_name[m])[0][k])
        if not std_dic.get(name):
            return '未查找到此人'
        else:
            score = input('输入你的英语、数学、专业课成绩，中间用空格隔开：')
            std_dic[name] = list(map(int, score.split(' ')))
            sum_score = std_dic[name][0] + std_dic[name][1] + std_dic[name][2]
            std_dic[name].append(sum_score)
    with open('学生信息数据库.txt', 'w', encoding='utf-8') as f:
            for j in std_dic.keys():
                f.write(j + '---' + str(std_dic[j][0]) + '---' + str(std_dic[j][1]) + '---' + str(std_dic[j][2]) +
                        '---' + str(std_dic[j][3]))
                f.write('\n')
            return '成功修改'


def total_std():
    save_std()
    total = len(std_dic.keys())
    return total


def order():
    order_std = {}
    list_order = []
    save_std()
    for i in std_dic.keys():
        order_std[i] = int(std_dic[i][3])
    tuple_order = sorted(order_std.items(), key=lambda x: x[1], reverse=True)
    for j in range(len(tuple_order)):
        list_order.append(tuple_order[j][0])
    with open('学生信息数据库.txt', 'w', encoding='utf-8') as f:
        for k in list_order:
            f.write(k + '---' + str(std_dic[k][0]) + '---' + str(std_dic[k][1]) + '---' + str(std_dic[k][2]) + '---'
                    + str(std_dic[k][3]))
            f.write('\n')
    return '成功排序'


def menu():
    print('+-----------------------------------------------------+')
    print('+                                                     +')
    print('+                      主菜单                         +')
    print('+                  0.退出数据库                       +')
    print('+                  1.插入学生记录                     +')
    print('+                  2.删除学生记录                     +')
    print('+                  3.修改学生记录                     +')
    print('+                  4.按总分降序排序                   +')
    print('+                  5.查找并显示学生的记录             +')
    print('+                  6.显示所有学生的记录               +')
    print('+                  7.统计数据库里学生人数             +')
    print('+                  8.显示主菜单                       +')
    print('+                                                     +')
    print('+-----------------------------------------------------+')


def main():
    menu()
    while True:
        num = input('输入你需要的功能的编号：')
        if num == '0':
            return print('退出程序')
        elif num == '1':
            print(std_msg())
        elif num == '2':
            print(del_record())
        elif num == '3':
            print(alter_record())
        elif num == '4':
            print(order())
        elif num == '5':
            print(show_msg())
        elif num == '6':
            show_all_msg()
        elif num == '7':
            print(total_std())
        elif num == '8':
            menu()


if __name__ == '__main__':
    main()
