# coding=utf-8

class Operator:
    level = 0 #层数
    m = 0 #该层的第m个操作
    def __init__(self,level,m):
        self.level=level
        self.m=m
#第0层为基本操作
#每一层的-1代表identity,0代表None