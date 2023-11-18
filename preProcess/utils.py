import os
import random

import networkx as nx
import pandas as pd


def create_graph():
    project_dir = os.path.dirname(os.getcwd())
    data_dir = '/home/lighthouse/rl_mtop/data'
    # 读取CSV文件
    data = pd.read_csv(data_dir + '/dis_CBD_twoPs_03_19.csv')
    print(data_dir)
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
        if node1 not in graph.nodes:
            graph.add_node(node1)
        if node2 not in graph.nodes:
            graph.add_node(node2)
        # 检查边是否已经存在于图中
        if not graph.has_edge(node1, node2):
            graph.add_edge(node1, node2, distance=dis)
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
        self.start_time = None
        self.Request = None
        self.idx = None
        self.money = None
        self.speed = speed
        self.pos = None


# 从CSV文件中导入请求
def import_requests_from_csv():
    project_dir = os.path.dirname(os.getcwd())
    data_dir = '/home/lighthouse/rl_mtop/data'
    file_path = data_dir + "/bay_vio_data_03_19.csv"
    requests = [[]]
    data = pd.read_csv(file_path)
    for row in data.itertuples(index=False):
        timestamp = row.RequestTime
        destination = row.aim_marker
        origin = row.street_marker
        if timestamp >= len(requests):
            requests.append([])
        request = Request(timestamp, destination, origin)
        requests[timestamp].append(request)
    return requests


def change_node_to_int(node):
    try:
        return int(node.replace("A", ""))
    except ValueError:
        return 0


if __name__ == '__main__':
    print(len(create_graph().nodes))
