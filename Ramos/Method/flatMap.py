# coding=utf-8

import copy
import numpy as np
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
from Test.print_saitama import Print_saitama
from DataStruct.edge import edge
def Decode(type, ch):
    res = 1
    same_channel_operators = [-1,2,4,5,8,9,10,11,12,13,14]
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
                branches.append(edge(target, j, Operation))
    # #打印各点的channels
    # print("各点的channels为：")
    # for i in range(len(f.channels)):
    #     print(i)
    #     print(f.channels[i])
    return


#将一个高阶算法解析成其对应的计算图
def transform(f, op, i, j):
    f.Map[i][j] = Operator(0, 0)  #切断原来的边

    new_size = f.size + op.size
    new_f = FlatOperatorMap(size=new_size)

    for x in range(new_size):
        for y in range(new_size):
            new_f.Map[x][y] = Operator(0, 0)
    for x in range(f.size):
        for y in range(f.size):
            new_f.Map[x][y] = f.Map[x][y]

    temp_size = f.size
    oMap = copy.deepcopy(op.Map)
    for x in range(op.size):
        for y in range(op.size):
            #若计算图中的边不是none，那么就把该边加入到扁平图中
            if oMap[x][y].m != 0:
                new_f.Map[temp_size + x][temp_size +y] = oMap[x][y]
                #如果是identity操作的话，改为基本型
                if oMap[x][y].m == -1:
                    new_f.Map[temp_size + x][temp_size + y].level = 0
            else:
                continue
    new_f.Map[i][temp_size] = Operator(0, -1)#前驱节点与子计算图起点用identity连接
    new_f.Map[new_size - 1][j] = Operator(0, -1)#后继节点与子计算图终点用identity连接

    f.size = new_size
    f.Map = copy.deepcopy(new_f.Map)
    return

def toFlatMap(g):
    global mainPath
    global branches
    # 初始化，传入最高层计算图

    topLevel = g.level
    f_size = g.operatorMaps[topLevel - 1][0].size
    f = FlatOperatorMap(size=f_size)
    f.Map = copy.deepcopy(g.operatorMaps[topLevel-1][0].Map)
    # 标记，是否已经解析成了扁平图
    flag = 1

    #最多解析topLevel-1轮
    for i in range(topLevel - 1):
        if (not flag):
            break

        flag = 0
        #高层方法展开
        for i in range(f.size):
            for j in range(f.size):
                if f.Map[i][j].m == -1 or f.Map[i][j].m == 0:
                    f.Map[i][j].level = 0
                    continue
                # 不同level间的identity与none是完全一致的
                # 不需要任何解析操作

                #找到了除基本方法以外的方法
                if f.Map[i][j].level != 0:
                    #依然存在高层方法，需要解析，flag置为1
                    flag = 1

                    op_level = f.Map[i][j].level
                    op_m = f.Map[i][j].m
                    op_Map = g.operatorMaps[op_level - 1][op_m - 1]#找到目标计算图
                    transform(f, op_Map, i, j)
                else:
                    continue
    # #用于打印flatmap的方法
    # Print_saitama(f,f.size)

    mainPath = []
    branches = []
    # for i in range(f.size):
    #     branches.append([])
    decodeChannel(f)
    # TODO
    # 添加方法的channels
    for branch in branches:
        branch.channel = f.channels[branch.fromIndex];
    # TODO
    # 添加方法的concat
    for i in range(len(branches)):
        # -1 is the mark which indicates that the branch remains unvisited
        if branches[i].isConcat > -1:
            continue
        # 0 indicates don't need concat
        branches[i].isConcat = 0
        for j in range(i + 1, (len(branches))):
            if branches[j].toIndex == branches[i].toIndex and branches[j].isConcat == -1:
                branches[j].isConcat = 1
            else:
                continue
    GlobalConfig.final_module = copy.deepcopy(branches)
    GlobalConfig.channels = copy.deepcopy(f.channels)
    # print(mainPath)
    # for i in range(len(branches)):
    #     print(branches[i].fromIndex, ' ', branches[i].toIndex, ' ', branches[i].operator)
    # for i in range(f.size):
    #     for j in range(len(branches[i])):
    #         print(branches[i][j].fromIndex,' ',branches[i][j].toIndex,' ',branches[i][j].operator)
    return f
    #注：输入类型为Genetype
    #输出类型为flatOperatorMap