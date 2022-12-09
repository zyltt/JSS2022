# coding=utf-8

from DataStruct.operation import Operator
#每个OperatorMap中的初始为identity链
class OperatorMap:
    Map=[]
    size=0
    def __init__(self,size):
        self.size=size
        self.Map=[]#防止调用构造函数带来的相互影响。
        for i in range(size):
            self.Map.append([])
            for j in range(size):
                if (j==i+1):
                    self.Map[i].append(Operator(0, -1))
                else:
                    self.Map[i].append(Operator(0, 0))
#调用方式：a=OperatorMap(size=6)