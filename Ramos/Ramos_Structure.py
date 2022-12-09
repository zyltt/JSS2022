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

    # TODO 修改为用于保存结果的文件目录
    result_csv = './Ramos_graph.csv'

    with open(result_csv, 'w', encoding= 'utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        data = ['id', '最长通路', '总边数', '最长通路长度与总边数的比例', '总节点数']
        csv_writer.writerow(data)

    # TODO 修改为result.csv所在目录

    file_path = './result.csv'
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        model_structrues = [row[10] for row in reader]
    del model_structrues[0]

    # # TODO 取最后K组数据
    # K = 100
    # model_structrues = model_structrues[len(model_structrues) - K:]

    Round = 0
    for model_structure in model_structrues:

        Round += 1

        edge_infos = model_structure.split("  ")
        del edge_infos[len(edge_infos) - 1]

        edges = []
        point_num = 0

        for edge_info in edge_infos:
            elements = edge_info.split(" ")
            from_id = 0
            to_id = 0
            op = 0
            for element in elements:
                key = int(element[element.find(":") + 1:])
                if element.__contains__("from"):
                    from_id = key
                    if key > point_num:
                        point_num = key
                if element.__contains__("operator"):
                    op = key
                else:
                    if element.__contains__("to"):
                        to_id = key
                        if key > point_num:
                            point_num = key
            this_edge = edge(from_id, to_id, op)
            edges.append(this_edge)


        point_num += 1
        total_edge = len(edges)

        f = FlatOperatorMap(size=point_num)
        for x in range(f.size):
            for y in range(f.size):
                f.Map[x][y] = Operator(0, 0)

        for each_edge in edges:
            this_i = each_edge.fromIndex
            this_j = each_edge.toIndex
            this_op = each_edge.operator
            f.Map[this_i][this_j] = Operator(0, this_op)

        in_degree = [0] * f.size
        for i in range(f.size):
            for j in range(f.size):
                if f.Map[i][j].m != 0:
                    in_degree[j] += 1

        ans = tuopu(f.Map, f.size, [0], 0, edges[len(edges) - 1].toIndex)
    #
        with open(result_csv, "a+", encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            data = [Round, ans, total_edge, ans / total_edge, point_num]
            csv_writer.writerow(data)

        print(model_structure)
        print(Round,  ": ", ans, " vs ", total_edge, " rate: ", ans / total_edge)
#             total += 1
#             if ans == last_layer + 1:
#                 all_count += 1
#             if ans >= ((last_layer + 1) / 3) * 2:
#                 half_count += 1
