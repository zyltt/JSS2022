import json
import os
import datetime
import copy
import os
import csv
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge

def Decode(type, ch):
    res = 1
    same_channel_operators = [-1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    if type in same_channel_operators:
        res = ch

    return res


def search_zero(in_degree, size):
    for i in range(size):
        if in_degree[i] == 0:
            return i
    return -1


def decodeChannel(f):
    global mainPath
    global branches
    #注：输入类型为flatOperaotrMap

    #先把f.chanels扩大
    f.channels = [0]*f.size
    f.channels[0] = 1
    in_degree = [0]*f.size
    for j in range(f.size):
        for i in range(f.size):
            if f.Map[i][j].m != 0:
                in_degree[j] += 1

    #最多拓扑f.size轮
    for times in range(f.size):
        # 找到入度为0的点
        target = search_zero(in_degree, f.size)
        if target < 0:
            print("Error! Circle exits!")
            return

        # mainPath.append(target + 1);
        # length = len(mainPath)
        # if length > 1:
        #     FromIndex = mainPath[length - 2] - 1
        #     ToIndex = target
        #     Operation = f.Map[FromIndex][ToIndex].m
        #     branches.append(edge(FromIndex,ToIndex,Operation))

            # for toIndex in range(f.size):
                # if toIndex == ToIndex:
                #     continue
                # if f.Map[FromIndex][toIndex].m != 0:
                #     Operation = f.Map[FromIndex][toIndex].m
                #     branches.append(edge(FromIndex, toIndex, Operation))


        in_degree[target] = -1
        for j in range(f.size):
            if f.Map[target][j].m != 0:

                # #用于引导和测试模型的专用语句 mark
                # if f.Map[target][j].m != 4:
                #     f.Map[target][j].m = 1;

                in_degree[j] -= 1
                f.channels[j] += Decode(f.Map[target][j].m, f.channels[target])
                Operation = f.Map[target][j].m
                branches.append(edge(target, j, Operation))
    # #打印各点的channels
    # print("各点的channels为：")
    # for i in range(len(f.channels)):
    #     print(i)
    #     print(f.channels[i])
    return

def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

if __name__ == '__main__':
    global mainPath
    global branches
    start_time = datetime.datetime.now()
    for dataset in ['random']:
    # for dataset in ['mnist']:
        result_csv = './muffin_result.csv'
        GlobalConfig.dataset = dataset

        with open(result_csv, 'w', encoding= 'utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            data = ['model_id', 'model_result']
            csv_writer.writerow(data)

        file_path = './Muffin_models/'
        for root, dirs, files in os.walk(file_path):
            for dir in dirs:
                if not os.path.exists(file_path + dir + '/models/model.json'):
                    continue
                InputPath = open(file_path + dir + '/models/model.json', encoding="utf-8")
                json_dic = json.load(InputPath)
                model = json_dic['model_structure']
                first_list = json_dic['input_id_list']
                if len(first_list) > 1:
                    print("非法格式，模型存在多个输入！")
                    continue
                last_list = json_dic['output_id_list']
                if len(last_list) > 1:
                    print("非法格式，模型存在多个输出！")
                    continue

                # TODO First_Layer 表示输入层
                # TODO Last_layer 表示最后的输出层
                first_layer = first_list[0]
                last_layer = last_list[0]

                f = FlatOperatorMap(size=last_layer + 1)
                for x in range(f.size):
                    for y in range(f.size):
                        f.Map[x][y] = Operator(0, 0)

                model = json_dic['model_structure']
                for layer in range(first_layer, last_layer + 1):
                    id = str(layer)
                    this_layer_inf = model[id]
                    from_ids = this_layer_inf['pre_layers']
                    to_id = layer
                    Type = this_layer_inf["type"]
                    if Type in ['concatenate', 'average', 'maximum', 'minimum', 'add', 'subtract', 'multiply', 'dot']:
                        if id != str(last_layer):
                            Type = 'concatenate'
                        else:
                            Type = 'identity'

                    if Type == "input_object" or Type == 'concatenate':
                        continue
                    for from_id in from_ids:
                        from_layer = model[str(from_id)]
                        if from_layer['type'] in ['concatenate', 'average', 'maximum', 'minimum', 'add', 'subtract',
                                                  'multiply', 'dot']:
                            for true_from_id in from_layer["pre_layers"]:
                                # f.Map[true_from_id][to_id] = Type
                                f.Map[true_from_id][to_id] = Operator(0, parse_Type(Type))
                        else:
                            # f.Map[from_id][to_id] = Type
                            f.Map[from_id][to_id] = Operator(0, parse_Type(Type))

                mainPath = []
                branches = []
                decodeChannel(f)
                for branch in branches:
                    branch.channel = f.channels[branch.fromIndex]
                GlobalConfig.final_module = copy.deepcopy(branches)
                GlobalConfig.channels = copy.deepcopy(f.channels)

                from Method.module_executor import exe_module
                torch_tf_max_diff, torch_tf_mean_diff, \
                torch_mindspore_max_diff, torch_mindspore_mean_diff, \
                tf_mindspore_max_diff, tf_mindspore_mean_diff, \
                avg_max_diff, avg_mean_diff = exe_module()

                with open(result_csv, "a+", encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f)
                    data = [f'{dataset}-{dir}', avg_max_diff]
                    csv_writer.writerow(data)
                print(dataset, "-", dir, ": ", avg_max_diff)
                end_time = datetime.datetime.now()
                duration = end_time - start_time
                second = duration.seconds
                if second >= 3600:
                    break