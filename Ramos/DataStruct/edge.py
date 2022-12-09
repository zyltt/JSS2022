# coding=utf-8

class edge:
    fromIndex = 0
    toIndex = 0
    # 为了方便格式转化，且涉及的操作仅为基本操作，故只保留操作号，不保留层号
    operator = 0
    channel = 0
    isConcat = -1
    def __init__(self, FromIndex, ToIndex, Operator):
        self.fromIndex = FromIndex
        self.toIndex = ToIndex
        self.operator = Operator
        self.channel = 0
        self.isConcat = -1