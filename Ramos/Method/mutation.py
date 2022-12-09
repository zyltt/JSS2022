# coding=utf-8

from DataStruct.genetype import Genetype
from DataStruct.globalConfig import GlobalConfig
from DataStruct import genetype
from DataStruct import genetypeQueue
from DataStruct import operatorMap
import copy
import random
import math
import numpy as np
from Test.print_saitama import Print_saitama

global map_size
global single_source_and_sink
global color
global found

def mutation(g):
    #selection
    global map_size
    global single_source_and_sink

    level = GlobalConfig.L

    #确定在哪一层突变（注意作为下标时要-1）
    l = math.ceil(random.random()*level)
    maps = g.operatorMaps[l - 1]
    map_size = GlobalConfig.pointNum[l - 1]
    num = GlobalConfig.operatorNum[l - 1]
    #确定在哪一个计算图突变
    id = math.ceil(random.random()*num)
    map = maps[id - 1].Map

    # mutation
    acyclic_and_connected = False
    single = False
    while not(acyclic_and_connected and single):
        #确定与突变关联的两个节点
        i=0
        j=0
        while i == j:
            i = math.floor(random.random() * map_size)
            j = math.floor(random.random() * map_size)
        if i > j:
            i,j = swap(i,j)
        # while i == j:
        #     j = math.floor(random.random() * map_size)

        old_Lvl = map[i][j].level
        old_m = map[i][j].m

        # 如果需要满足链的要求，就用这个版本
        # if i + 1 < j:
        #     ran_Lvl = 0
        # else:
        #     ran_model = random.random();
        #     if ran_model <= GlobalConfig.basicProp:
        #         ran_Lvl = 0
        #     else:
        #         ran_Lvl = l - 1

        # TODO basicProp
        ran_model = random.random();
        if ran_model <= GlobalConfig.basicProp:
            ran_Lvl = 0
        else:
            ran_Lvl = l - 1
        #ran_Lvl = math.floor(random.random() * l)

        #可能会突变到基本操作，所以operationNum会越界
        # if ran_Lvl > 0:
        #     total_num = GlobalConfig.operatorNum[ran_Lvl - 1]
        # else:
        #     total_num = 7 #除了none和identity外的7种基本操作
        # ran_m = math.ceil(random.random() * total_num + 2) - 2#包括0和-1，级identity和none

        # random with weights
        ran_m = 0
        if ran_Lvl == 0:
            total = 0
            weights = GlobalConfig.basicWeights
            chart = [0] * len(weights)
            for p in range(len(weights)):
                total += weights[p]
                chart[p] = total

            thisRan = random.random() * total
            for p in range(len(weights)):
                if thisRan <= chart[p]:
                    ran_m = p - 1
                    break

            while ran_Lvl == old_Lvl and ran_m == old_m:
                thisRan = random.random() * total
                for p in range(len(weights)):
                    if thisRan < chart[p]:
                        ran_m = p - 1
                        break
        else:
            total = 0
            weights = g.weights[l - 2]
            chart = [0] * len(weights)
            for p in range(len(weights)):
                total += weights[p]
                chart[p] = total

            thisRan = random.random() * total
            for p in range(len(weights)):
                if thisRan <= chart[p]:
                    ran_m = p + 1
                    break

            while ran_Lvl == old_Lvl and ran_m == old_m:
                thisRan = random.random() * total
                for p in range(len(weights)):
                    if thisRan < chart[p]:
                        ran_m = p + 1
                        break


        ###为了debug的方法修改
        # if ran_Lvl==0 and ran_m==3:
        #     ran_m=-1
        # if ran_Lvl==0 and ran_m==2:
        #     ran_m=-1
        #
        # while ran_Lvl == old_Lvl and ran_m == old_m:
        #     ran_m = math.ceil(random.random() * total_num + 2) - 2
        #
        ##保证成链的限制
        # if ran_m == 0 and i == j - 1:
        #     continue;

        #-1和0统一成基本操作
        if ran_m == 0 or ran_m == -1:
            ran_Lvl = 0

        map[i][j].level = ran_Lvl
        map[i][j].m = ran_m

        #judge acyclic , connective and connected
        single = judge_single_source_and_sink(map)
        if single:
            acyclic_and_connected = judge_acyclic_and_connected(map)
            #print(acyclic_and_connected)
            #print(single)
            if acyclic_and_connected and single:
                #mutation success
                #print(newg.operatorMaps[l])
                # print("mutation success!")
                #Print_saitama(maps[id - 1], map_size)

                #TODO record mutation
                g.mutateL = ran_Lvl
                g.mutateM = ran_m
                return

        map[i][j].level = old_Lvl
        map[i][j].m = old_m
    return

def swap(i, j):
    return j, i;
#utils
# 注意到由于子图的联通性和是否有环都已经判断过，如果子图无环，展开子图后也不会新增环
# 所以我们只需要判读整个大图抽象结构上的性质，不需要再展开子图了
def judge_acyclic_and_connected(map):
    #use dfs to judge
    #this is dfs wrapper
    global acyclic
    global color
    global map_size
    global connected

    connected = False

    # color=np.zeros(map_size) # 0:unseen,1:visiting,2:visited
    # acyclic = False
    # findcyclic(0,0,map)
    # if acyclic:
    #     return False

    color = np.zeros(map_size)  # 0:unseen,1:visi
    # ting,2:visited
    dfs(0,0,map)

    # if found:
    #     acyclic = False
    #     #print("Has Cycle")
    # else:
    #     acyclic = True
    #     #print("No Cycle")

    #if connected:
        #print("Connected")
    #else:
        #print("DisConnected")

    if connected:
        return True
    else:
        return False

def findcyclic(dep, node, map):
    global connected
    global acyclic
    global map_size

    if acyclic:
        return

    color[node] = 1
    for i in range(map_size):
        if map[node][i].m != 0:
            if color[i] == 1:
                acyclic = True
                return
            else:
                findcyclic(dep + 1, i, map)
        if acyclic:
            return
    return

def dfs(dep, node, map):
    # global found
    global color
    global connected
    global map_size

    #染色
    color[node] = 1

    if np.all(color):
        #此时连通
        connected = True

    for i in range(map_size):
        if map[node][i].m != 0:
            # if color[i] == 1:    #if has circle
            #     found = True     #发现有环
            #     return
            # else:
            if color[i] == 2:
                continue
            else:
                dfs(dep+1, i, map)
    #退栈的时候要标注已经访问完成，要不然会有误判
    color[node] = 2
    return

def judge_single_source_and_sink(map):
    global map_size
    #保证0为起点且n为终点
    for i in range(map_size):
        if map[map_size - 1][i].m != 0:#发现n有出边
            return False
        if map[i][0].m != 0:#发现0有入边
            return False

    #寻找是否存在其它起点或终点

    in_degree = [0] * map_size#记录入度的数组
    out_degree = [0] * map_size#记录出度的数组

    for i in range(map_size):
        for j in range(map_size):
            if map[i][j].m != 0:
                in_degree[j] += 1
                out_degree[i] += 1

    for i in range(map_size):
        #发现不为1的起点
        if in_degree[i] == 0 and i != 0:
            return False
        #发现不为n的终点
        if out_degree[i] == 0 and i != map_size - 1:
            return False

    return True
