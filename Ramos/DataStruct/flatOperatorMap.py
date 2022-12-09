# coding=utf-8

from DataStruct.operatorMap import OperatorMap
class FlatOperatorMap(OperatorMap):
    channels=[]
    def __init__(self,size):
        OperatorMap.__init__(self,size=size)
        self.channels=[0]*size
#调用方式：b=FlatOperatorMap(size=6)


