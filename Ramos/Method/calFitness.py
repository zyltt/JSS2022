# coding=utf-8

from .flatMap import toFlatMap
def calFitness(g):
    #todo
    toFlatMap(g)
    #一定要用到的时候再import，否则会初始化错误的值
    from Method.module_executor import exe_module
    torch_tf_max_diff,torch_tf_mean_diff,\
    torch_mindspore_max_diff,torch_mindspore_mean_diff,\
    tf_mindspore_max_diff,tf_mindspore_mean_diff,\
    avg_max_diff,avg_mean_diff=exe_module()
    return torch_tf_max_diff,torch_tf_mean_diff,torch_mindspore_max_diff,torch_mindspore_mean_diff,tf_mindspore_max_diff,tf_mindspore_mean_diff,avg_max_diff,avg_mean_diff
