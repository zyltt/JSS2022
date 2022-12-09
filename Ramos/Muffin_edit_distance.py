import networkx as nx
import json
import os
import csv
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig

# Calculate the average edit distance, that is, the average edit distance between two adjacent graphs


def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def edge_match(edge1, edge2):
    if edge1['operator'] == edge2['operator']:
        return True
    else:
        return False


if __name__ == '__main__':
    Maps = []
    file_path = './cifar10_output/'
    round = 0
    for root, dirs, files in os.walk(file_path):
        for dir in dirs:
            if not os.path.exists(file_path + dir + '\\models\\model.json'):
                print("system processing")
                continue
            InputPath = open(file_path + dir + '\\models\\model.json', encoding="utf-8")
            round = round + 1
            print('model',round,'is loading')
            json_dic = json.load(InputPath)
            model = json_dic['model_structure']
            first_list = json_dic['input_id_list']
            if len(first_list) > 1:
                print("too many inputs!")
                continue
            last_list = json_dic['output_id_list']
            if len(last_list) > 1:
                print("too many outputs")
                continue

            # TODO First_Layer Input Layer
            # TODO Last_layer Output Layer
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
            G = nx.DiGraph()
            for i in range(f.size):
                for j in range(f.size):
                    if f.Map[i][j].m != 0:
                        G.add_edge(v_of_edge = i, u_of_edge= j, operator = f.Map[i][j].m)
            Maps.append(G)
    edit_distances = []
    for i in range(0, len(Maps) - 1):
        j = i + 1
        print("from", i, "to", j, ":")
        # delete too large maps
        if Maps[i].number_of_edges() > 100:
            print("This Map is too large, delete it.")
            continue
        distance = nx.graph_edit_distance(Maps[i], Maps[j], edge_match=edge_match, timeout=1)
        print(distance)
        if distance != None:
            edit_distances.append(distance)
    average_edit_distance = sum(edit_distances) / len(edit_distances)
    max_edit_distance = max(edit_distances)
    min_edit_distance = min(edit_distances)
    print("The average edit distance is:")
    print(average_edit_distance)
    print("The max edit distance is:")
    print(max_edit_distance)
    print("The min edit distance is:")
    print(min_edit_distance)
