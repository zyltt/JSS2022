# coding=utf-8

from DataStruct.operatorMap import OperatorMap
from DataStruct.globalConfig import GlobalConfig
from DataStruct.operation import Operator
class Genetype:
    level=0
    fitness=0
    operatorMaps=[]
    weights=[]
    mutateL = -2
    mutateM = -2
    def __init__(self,level):
        self.operatorMaps=[]#为了消除多次调用时的相互影响
        self.level=level
        self.fitness=0
        self.mutateL = -2
        self.mutateM = -2
        for i in range(level):
            self.operatorMaps.append([])
            for j in range(GlobalConfig.operatorNum[i]):
                temp=OperatorMap(size=GlobalConfig.pointNum[i])
                self.operatorMaps[i].append(temp)

        for i in range(level):
            self.weights.append([])
            for j in range(GlobalConfig.operatorNum[i]):
                self.weights[i].append(1)

#调用方式：b=Genetype(level=GlobalConfig.L)