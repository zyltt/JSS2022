# coding=utf-8

class Population:
    size=0
    genetypes=[]
    def __init__(self):
        self.size=0
        self.genetypes=[]
    def append(self,g):
        self.genetypes.append(g)
        self.size=self.size+1
#创建和加入种群示例
# b=Genetype(level=2)
# p=Population()
# p.append(b)
