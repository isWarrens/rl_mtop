import os
import random

import networkx as nx
import pandas as pd
import csv
import tempfile
import shutil


def create_graph():
    project_dir = os.path.dirname(os.getcwd())
    data_dir = project_dir + '/rl_mtop/data'
    # 读取CSV文件
    data = pd.read_csv(data_dir + '/dis_CBD_twoPs_03_19.csv')
    # 创建一个无向图
    graph = nx.Graph()

    # 添加节点和边
    for row in data.itertuples(index=False):
        dis = row.distance
        nodes = row.twoPs.split('_')
        node1 = nodes[0]
        node2 = nodes[1]
        try:
            int(node1.replace("A", ""))
            int(node2.replace("A", ""))
        except ValueError:
            continue
        # 检查节点是否已经存在于图中
        if int(node1.replace("A", "")) not in graph.nodes:
            graph.add_node(int(node1.replace("A", "")))
        if int(node2.replace("A", "")) not in graph.nodes:
            graph.add_node(int(node2.replace("A", "")))
        # 检查边是否已经存在于图中
        if not graph.has_edge(int(node1.replace("A", "")), int(node2.replace("A", ""))):
            graph.add_edge(int(node1.replace("A", "")), int(node2.replace("A", "")), distance=dis)
    return graph


def choose_random_node(graph):
    nodes = list(graph.nodes)
    random_node = random.choice(nodes)
    return random_node


import pandas as pd


# 定义请求结构
class Request:
    def __init__(self, timestamp, destination, origin):
        self.timestamp = timestamp
        self.destination = destination
        self.origin = origin
        self.state = 0


class Driver:
    def __init__(self, speed):
        self.on_road = None
        self.start_time = 0
        self.Request = None
        self.idx = None
        self.money = None
        self.speed = speed
        self.pos = None


# 从CSV文件中导入请求
def import_requests_from_csv():
    project_dir = os.path.dirname(os.getcwd())
    data_dir = project_dir + '/rl_mtop/data'
    file_path = data_dir + "/bay_vio_data_03_19.csv"
    requests = [[]]
    data = pd.read_csv(file_path)
    for row in data.itertuples(index=False):
        timestamp = row.RequestTime
        destination = change_node_to_int(row.aim_marker)
        origin = change_node_to_int(row.street_marker)
        if timestamp >= len(requests):
            requests.append([])
        request = Request(timestamp, destination, origin)
        if request != origin:
            requests[timestamp].append(request)
    return requests


def change_node_to_int(node):
    try:
        return int(node.replace("A", ""))
    except ValueError:
        return 0

def change_csv():
    project_dir = os.path.dirname(os.getcwd())
    data_dir = project_dir + '/rl_mtop/data'+ '/bay_vio_data_03_19.csv'

    # 读取原始CSV文件的数据并筛选
    with open(data_dir, 'r', newline='') as input_file:
        reader = csv.reader(input_file)
        # 创建一个临时文件来保存筛选后的数据
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

        writer = csv.writer(temp_file)
        # 读取并写入第一行
        header = next(reader)
        writer.writerow(header)        # 跳过第一行
        next(reader)
        for row in reader:
            nodes = row[1].split('_')
            node1 = nodes[0]
            node2 = nodes[1]
            try:
                int(node1.replace("A", ""))
                int(node2.replace("A", ""))
            except ValueError:
                continue

            if int(node1.replace("A", "")) > 100 | int(node2.replace("A", "")) > 100:
                continue

            writer.writerow(row)

    # 关闭临时文件
    temp_file.close()

    # 覆盖原始文件
    shutil.move(temp_file.name, data_dir)



if __name__ == '__main__':
    print(len(create_graph().nodes))
