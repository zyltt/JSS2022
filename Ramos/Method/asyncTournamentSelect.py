# coding=utf-8
import copy

from DataStruct.genetype import Genetype
from DataStruct.operatorMap import OperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
import math
import random

def asyncTournamentSelect(p):
    # TODO
    # 按照论文原文的实现, 锦标赛大小被设置为人口大小的5%
    pressure = 0.05
    K = GlobalConfig.k;
    sample_set_size = math.floor(p.size * pressure)  #锦标赛大小
    sample_set = random.sample(p.genetypes, sample_set_size * K); #随机选取，保证没有重复
    candidate_set = [];
    for i in range(K):
        this_sample_set = sample_set[sample_set_size * i: sample_set_size * (i + 1)]; #每次取一个集合
        this_sample_set.sort(key=lambda x:x.fitness, reverse=True);
        this_sample_set = this_sample_set[0:K];
        for this_genetype in this_sample_set:
            genetype = copy.deepcopy(this_genetype);
            candidate_set.append(genetype);

    # best_fitness_individual_index = find_best_fitness_index(sample_set)  # 锦标赛选择
    # best_fitness_individual = sample_set[best_fitness_individual_index]
    # return cloneGenetype(best_fitness_individual)  # 深复制
    candidate_set.sort(key=lambda y:y.fitness, reverse=True);
    return candidate_set[0:K]


def find_best_fitness_index(genetypes_set):
    maxIndex = 0
    length = len(genetypes_set)
    for i in range(length):
        if genetypes_set[i].fitness > genetypes_set[maxIndex].fitness:
            maxIndex = i
    return maxIndex


# def cloneGenetype(genetype):
#     res = Genetype(genetype.level)
#     res.level = genetype.level
#     res.fitness = genetype.fitness
#     for i in range(len(genetype.operatorMaps)):
#         for j in range(len(genetype.operatorMaps[i])):
#             res.operatorMaps[i][j] = cloneOperatorMap(genetype.operatorMaps[i][j])
#     return res
#
#
# def cloneOperatorMap(operatormap):
#     res = OperatorMap(operatormap.size)
#     for i in range(operatormap.size):
#         for j in range(operatormap.size):
#             res.Map[i][j] = cloneOperator(operatormap.Map[i][j])
#     return res
#
#
# def cloneOperator(operator):
#     return Operator(operator.level, operator.m)
