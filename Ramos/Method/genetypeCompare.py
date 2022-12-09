# coding=utf-8

from DataStruct.globalConfig import GlobalConfig

def compare(op1, op2):
    #比较两个计算图的节点个数是否一致
    #虽然我感觉肯定是一样的
    if op1.size != op2.size:
        return False

    #比较两个计算图是否完全一致
    total_size = op1.size
    for i in range(total_size):
        for j in range(total_size):
            if op1.Map[i][j].m != op2.Map[i][j].m:
                return False
            else:
                # 可能会出现属于不同层的none和identity操作，他们实际上是一样的
                if op1.Map[i][j].m == 0 or op1.Map[i][j].m == -1:
                    continue
                else:
                    if op1.Map[i][j].level != op2.Map[i][j].level:
                        return False
    return True

def genetypeCompare(g1, g2):
    #todo

    # 依次比较每个计算图
    for i in range(GlobalConfig.L):
        #total = len(g1.operatorMaps[i])
        total = GlobalConfig.operatorNum[i]
        for j in range(total):
            if not compare(g1.operatorMaps[i][j], g2.operatorMaps[i][j]):
                return False
    return True