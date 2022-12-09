# coding=utf-8
import traceback

from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
from DataStruct.globalConfig import GlobalConfig
from DataStruct.worker import Worker
from DataStruct.controller import Controller
from Method.initialize import initialize
from Method.util import getFinalModule_in_str,getChannels_in_str
import csv

def globalInit():
    # step1:配置globalConfig
    print("正在初始化globalConfig")
    out = open(file='./' + 'result.csv' , mode='w', newline='')
    writer = csv.writer(out,delimiter = ",")
    GlobalConfig.N = 0
    GlobalConfig.flatOperatorMaps = []
    GlobalConfig.resGenetype = []
    GlobalConfig.P = Population()
    GlobalConfig.Q = GenetypeQueue()
    GlobalConfig.final_module = []
    GlobalConfig.channels = []
    GlobalConfig.writer = writer
    writer.writerow(["No","torch_tf_max_diff","torch_tf_mean_diff",
                    "torch_mindspore_max_diff","torch_mindspore_mean_diff",
                     "tf_mindspore_max_diff","tf_mindspore_mean_diff",
                     "avg_max_diff","avg_mean_diff","channels","model","fail_time"])



globalInit()
print("正在初始化种群")
initialize(GlobalConfig.P)
print("种群初始化完成")
print("开始构建controller节点")
controller = Controller()
print("controller节点构建完成")
print("开始构建worker节点")
worker = Worker()
print("worker节点构建完成")

#主流程
t = 0
print("开始进行突变")
while(t < GlobalConfig.maxMutateTime):
    controller.excute()
    try:
        torch_tf_max_diff,torch_tf_mean_diff,torch_mindspore_max_diff,torch_mindspore_mean_diff,tf_mindspore_max_diff,tf_mindspore_mean_diff,avg_max_diff,avg_mean_diff = worker.excute()
        print("第" + str(t) + "轮已经完成")
        GlobalConfig.writer.writerow([str(t),str(torch_tf_max_diff),
                                      str(torch_tf_mean_diff),
                                      str(torch_mindspore_max_diff),
                                      str(torch_mindspore_mean_diff),
                                      str(tf_mindspore_max_diff),
                                      str(tf_mindspore_mean_diff),
                                      str(avg_max_diff),
                                      str(avg_mean_diff),
                                      getChannels_in_str(),
                                      getFinalModule_in_str(),
                                      str(GlobalConfig.fail_time)])
    except Exception as e:
        print("本轮突变失败")
        GlobalConfig.fail_time += 1
        print(traceback.format_exc())
    t = t + 1

#最后的筛选
while(len(GlobalConfig.resGenetype) < GlobalConfig.resultNum):
    controller.excute()
    thisg=GlobalConfig.Q.pop()
    GlobalConfig.resGenetype.append(thisg)