import copy
import csv
import os

from DataStruct.globalConfig import GlobalConfig
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from Method.module_executor import exe_module

def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def Decode(type, ch):
    res = 1
    same_channel_operators = [-1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    if type in same_channel_operators:
        res = ch

    return res

def search_zero(in_degree, size):
    for i in range(size):
        if in_degree[i] == 0:
            return i
    return -1

def decodeChannel(f):
    global mainPath
    global branches
    #注：输入类型为flatOperaotrMap

    #先把f.chanels扩大
    f.channels = [0]*f.size
    f.channels[0] = 1
    in_degree = [0]*f.size
    for j in range(f.size):
        for i in range(f.size):
            if f.Map[i][j].m != 0:
                in_degree[j] += 1

    #最多拓扑f.size轮
    for times in range(f.size):
        # 找到入度为0的点
        target = search_zero(in_degree, f.size)
        if target < 0:
            print("Error! Circle exits!")
            return

        # mainPath.append(target + 1);
        # length = len(mainPath)
        # if length > 1:
        #     FromIndex = mainPath[length - 2] - 1
        #     ToIndex = target
        #     Operation = f.Map[FromIndex][ToIndex].m
        #     branches.append(edge(FromIndex,ToIndex,Operation))

            # for toIndex in range(f.size):
                # if toIndex == ToIndex:
                #     continue
                # if f.Map[FromIndex][toIndex].m != 0:
                #     Operation = f.Map[FromIndex][toIndex].m
                #     branches.append(edge(FromIndex, toIndex, Operation))


        in_degree[target] = -1
        for j in range(f.size):
            if f.Map[target][j].m != 0:

                # #用于引导和测试模型的专用语句 mark
                # if f.Map[target][j].m != 4:
                #     f.Map[target][j].m = 1;

                in_degree[j] -= 1
                f.channels[j] += Decode(f.Map[target][j].m, f.channels[target])
                Operation = f.Map[target][j].m
    # #打印各点的channels
    # print("各点的channels为：")
    # for i in range(len(f.channels)):
    #     print(i)
    #     print(f.channels[i])
    return

class edge:
    fromIndex = 0
    toIndex = 0
    # 为了方便格式转化，且涉及的操作仅为基本操作，故只保留操作号，不保留层号
    operator = 0
    index = ""
    def __init__(self, FromIndex, ToIndex, Operator):
        self.fromIndex = FromIndex
        self.toIndex = ToIndex
        self.operator = Operator

GlobalConfig.dataset = 'mnist'
f1 = open('./lemon_model.csv', encoding = 'utf-8')
model_num = 0
while True:
    print(model_num)
    model_num += 1
    this_model_str = f1.readline()
    if this_model_str == "":
        break
    operators = this_model_str.split(" ")
    operators[0] = operators[0][1:]
    node_num = int(operators[-2].split(',')[1][3:])
    this_model = FlatOperatorMap(size=node_num+1)
    final_model = []
    for x in range(this_model.size):
        for y in range(this_model.size):
            this_model.Map[x][y] = Operator(0, 0)
    for each_operator in operators:
        if each_operator =='"\n':
            continue
        eachstr = each_operator.split(',')
        fromIndex = int(eachstr[0][5:])
        toIndex = int(eachstr[1][3:])
        type = parse_Type(eachstr[2][9:])
        this_model.Map[fromIndex][toIndex] = Operator(0,type)
        final_model.append(edge(FromIndex=fromIndex,ToIndex=toIndex,Operator=type))
    decodeChannel(this_model)
    GlobalConfig.final_module = copy.deepcopy(final_model)
    GlobalConfig.channels = copy.deepcopy(this_model.channels)

    from Method.module_executor import exe_module

    torch_tf_max_diff, torch_tf_mean_diff, \
    torch_mindspore_max_diff, torch_mindspore_mean_diff, \
    tf_mindspore_max_diff, tf_mindspore_mean_diff, \
    avg_max_diff, avg_mean_diff = exe_module()

    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    result_csv = './lemon_result.csv'

    with open(result_csv, "a+", encoding='utf-8', newline='') as f2:
        csv_writer = csv.writer(f2)
        data = [avg_max_diff]
        csv_writer.writerow(data)