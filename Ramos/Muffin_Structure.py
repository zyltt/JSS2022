import json
import os

import copy
import os
import csv
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge

def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def tuopu(Map, size, qidian, dep, target):
    global in_degree

    if qidian[0] == target:
        return dep
    else:
        this_list = []
        for i in qidian:
            for j in range(size):
                if f.Map[i][j].m != 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        this_list.append(j)

        if len(this_list) == 0:
            return -1
        else:
            return tuopu(Map, size, this_list, dep + 1, target)


if __name__ == '__main__':
    global in_degree
    total = 0
    all_count = 0
    half_count = 0

    for dataset in ['random']:
    # for dataset in ['fashion_mnist']:
        result_csv = './Muffin_graph.csv'

        with open(result_csv, 'w', encoding= 'utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            data = ['model_id', '最长通路', '总边数', '比例', '总节点数']
            csv_writer.writerow(data)

        file_path = './cifar10_output/'
        for root, dirs, files in os.walk(file_path):
            for dir in dirs:
                # if dataset == 'cifar10' and dir == '000101':
                #     break

                if not os.path.exists(file_path + dir + '\\models\\model.json'):
                    continue
                InputPath = open(file_path + dir + '\\models\\model.json', encoding="utf-8")
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

                point_num = last_layer - first_layer + 1

                total_edge = 0
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
                            point_num -= 1
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
                                total_edge += 1
                        else:
                            # f.Map[from_id][to_id] = Type
                            f.Map[from_id][to_id] = Operator(0, parse_Type(Type))
                            total_edge += 1

                in_degree = [0] * f.size
                for i in range(f.size):
                    for j in range(f.size):
                        if f.Map[i][j].m != 0:
                            in_degree[j] += 1

                ans = tuopu(f.Map, f.size, [0], 0, last_layer)

                with open(result_csv, "a+", encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f)
                    data = [f'{dataset}-{dir}', ans, total_edge, ans / total_edge, point_num]
                    csv_writer.writerow(data)

                print(dataset, "-", dir, ": ", ans, " vs ", total_edge, " rate: ", ans / total_edge)
    #             total += 1
    #             if ans == last_layer + 1:
    #                 all_count += 1
    #             if ans >= ((last_layer + 1) / 3) * 2:
    #                 half_count += 1
    # print(all_count, " vs ", half_count, " vs ", total)