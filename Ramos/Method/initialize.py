# coding=utf-8

import copy

from DataStruct.population import Population
from DataStruct.genetype import Genetype
from DataStruct.globalConfig import GlobalConfig
from DataStruct.operatorMap import OperatorMap
from DataStruct.operation import Operator
from DataStruct.flatOperatorMap import FlatOperatorMap
from .mutation import mutation
from .genetypeCompare import genetypeCompare
from DataStruct.globalConfig import GlobalConfig
from .flatMap import toFlatMap
from Test.print_saitama import Print_saitama


def check(g, p):
    for i in range(p.size):
        check_g = p.genetypes[i]
        if genetypeCompare(g, check_g):
            return False
        else:
            continue
    return True


def initialize(p):
    #todo
    #直接对p进行修改，没有返回值。
    #p代表population

    geno = Genetype(level=GlobalConfig.L)
    #g = copy.deepcopy(geno)
    #g.operatorMaps[3][0].Map[1][2] = Operator(3,2)
    #g.operatorMaps[2][1].Map[0][1] = Operator(2,1)
    #g.operatorMaps[1][0].Map[0][1] = Operator(1,3)
    #g.operatorMaps[0][2].Map[0][1] = Operator(0,2)

    #Print_saitama(g.operatorMaps[3][0], g.operatorMaps[3][0].size)
    #Print_saitama(g.operatorMaps[2][1], g.operatorMaps[2][1].size)

    #thisF = toFlatMap(g)
    #Print_saitama(geno.operatorMaps[2][1], g.operatorMaps[2][0].size)
    #Print_saitama(g.operatorMaps[1][1], g.operatorMaps[1][1].size)
    #thisF = toFlatMap(g)
    #Print_saitama(thisF,thisF.size)

    #Print_saitama(g.operatorMaps[1][2], g.operatorMaps[1][2].size)
    #print("::::::")
    #Print_saitama(geno.operatorMaps[1][2],geno.operatorMaps[1][2].size)

    #if genetypeCompare(g,geno):
    #    print(True)
    #else:
    #   print(False)

    times = 0
    g = copy.deepcopy(geno)
    while times < GlobalConfig.initMutateTime:
        #if check(g,p):
        #    print("True")
        #else:
        #    print("Flase")

        #如果g与population中的任何基因型都不相同，则check为True
        while not check(g,p):
            #注意是单次突变,所以要随时回档
            g = copy.deepcopy(geno)
            mutation(g)
            #突变次数计数+1
            times+=1
            if times >= GlobalConfig.initMutateTime:
                break

        #退出条件不同
        if check(g,p):
            p.genetypes.append(g)
            p.size+=1

    return

