# coding=utf-8

import numpy as np
from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
class GlobalConfig:
    fail_time = 0 # 失败构建模型的轮次。
    N = 0  # 现有种群规模
    L = 2  # 层数
    operatorNum = np.array([4,1])  # 每层计算图的个数
    pointNum = [4,4]  # 每层的节点数
    c0 = 3  # 初始通道数
    flatOperatorMaps = []  # 还原出的所有扁平计算图
    resultNum = 3  # 需要搜索出的重复结构数目
    resGenetype = []  # 搜索出的genentype集合
    maxMutateTime = 10000 # 最大突变次数
    P = Population()  # 初始种群
    Q = GenetypeQueue()  # controller选出的队列
    error_cal_mode = "max"#误差计算方式（只有max）
    initMutateTime = 200 # 初始化操作中执行突变的最大次数
    final_module = [] # 扁平图中节点之间的所有算子。
    channels = [] # 各节点的通道数。
    dataset = 'imagenet'#所使用的数据集，包含random,cifar10,mnist,fashion_mnist,imagenet,sinewave and price.共七种。
    h = 48 # featureMap的高度
    w = 48 # featureMap的宽度
    batch = 10 # 批次
    k = 1 # K锦标赛的阶数
    mode = 2 #反馈方式。0：只对基本算子反馈。1：只对复合算子反馈。2：两个反馈。
    writer = None # csv书写器
    basicOps = ['identity','None','1*1','depthwise_conv2D','separable_conv2D','max_pooling2D',
                'average_pooling2D','conv2D','conv2D_transpose','ReLU','sigmoid','tanh','leakyReLU',
                'PReLU','ELU','batch_normalization']#基本算子
    basicWeights = [1]*len(basicOps) # 基本操作的权重(包含-1和0)
    basicProp = 0.2 #随机到基本操作的概率