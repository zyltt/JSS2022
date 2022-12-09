# coding=utf-8
import random

from DataStruct.globalConfig import GlobalConfig
class Worker:
    a=0
    #todo
    def __init__(self):
        return
    def excute(self):
        g=GlobalConfig.Q.pop()
        #一定要运行的时候再import
        from Method.calFitness import calFitness
        torch_tf_max_diff,torch_tf_mean_diff,torch_mindspore_max_diff,torch_mindspore_mean_diff,tf_mindspore_max_diff,tf_mindspore_mean_diff,avg_max_diff,avg_mean_diff=calFitness(g)
        maxFitness = avg_max_diff
        meanFitness = avg_mean_diff
        #TODO thisfitness
        thisFitness = 0
        if GlobalConfig.error_cal_mode=="max":
            thisFitness=maxFitness
            print("本轮误差为"+str(maxFitness))
        else:
            thisFitness=meanFitness
            print("本轮误差为"+str(meanFitness))

        #TODO feedback
        #基本算子的反馈
        if g.mutateL == 0 and GlobalConfig.mode != 1:
            # Handle with Operator 0, Operator 0 is meaningless
            if g.mutateM != 0:
                GlobalConfig.basicWeights[g.mutateM + 1] += thisFitness - g.fitness
                if GlobalConfig.basicWeights[g.mutateM + 1] < 1e-6:
                    GlobalConfig.basicWeights[g.mutateM + 1] = 1e-6
                # Give a new weight to Operator 0
                total = 0
                for i in range(len(GlobalConfig.basicWeights)):
                    if i == 1:
                        continue
                    total += GlobalConfig.basicWeights[i]
                # TODO 删边的概率占基本操作整体概率的1/n
                GlobalConfig.basicWeights[1] = total / (len(GlobalConfig.basicOps) - 1)
        #复合算子的反馈
        if g.mutateL > 0 and GlobalConfig.mode != 0:
            g.weights[g.mutateL - 1][g.mutateM - 1] += thisFitness - g.fitness
            if g.weights[g.mutateL - 1][g.mutateM - 1] < 1e-6:
                g.weights[g.mutateL - 1][g.mutateM - 1] = 1e-6

        g.fitness = thisFitness
        g.mutateM = -2
        g.mutateL = -2
        GlobalConfig.P.append(g)
        return torch_tf_max_diff,torch_tf_mean_diff,torch_mindspore_max_diff,torch_mindspore_mean_diff,tf_mindspore_max_diff,tf_mindspore_mean_diff,avg_max_diff,avg_mean_diff