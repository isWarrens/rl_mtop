import random
import sqlite3
import os
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
from heapq import heappush, heappop, nsmallest
from tqdm import tqdm

from matplotlib import animation
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete
import hashlib
import json
from area import AreaInfo
from preProcess.utils import create_graph
from preProcess.utils import import_requests_from_csv
from preProcess.utils import Driver
from preProcess.utils import choose_random_node


class TopEnvironment(Environment):
    """
    Implementation of the travelling officer environment.
    """

    FREE = 0
    OCCUPIED = 1

    def __init__(self, gamma, drivers_num=0, speed=5000., observation=None, start_time=None, timestep=1):

        self.train_days = [39]
        self.drivers = []
        for i in range(drivers_num):
            self.drivers.append(Driver(0))
        # 初始化司机
        for idx, driver in enumerate(self.drivers):
            driver.on_road = 0
            driver.idx = idx
            driver.money = 0
            driver.speed = speed

        self.observation = observation
        self.events = None
        self.event_idx = None
        self.time = None
        self.done = False
        self.start_time = start_time
        self.timestep = timestep
        # 创建地图
        self.graph = create_graph()
        # 导入所有请求
        self.all_requests = import_requests_from_csv()
        # 当前时间下的所有请求
        self.requests = []
        # 所有的点
        self.actions = tuple(self.graph.nodes)
        mdp_info = MDPInfo(Discrete(1), Discrete(len(self.actions)), gamma, np.inf)
        super().__init__(mdp_info)

    def close(self):
        pass

    def reset(self, state=None):
        # 初始化司机在任意位置
        for driver in self.drivers:
            driver.on_road = 0
            driver.money = 0
            driver.Request = None
            driver.pos = choose_random_node(self.graph)

        self.time = 0
        self.requests.extend(self.all_requests[0])
        self.done = False
        return self._state()

    def _state(self):
        return self.observation(self)

    # action是request[] action是一一对应的
    def step(self, action):
        # action把他变成司机->request的形式传入step
        node_idx = action[0]
        select_actions = []
        for r in self.requests:
            if r.destination == node_idx:
                select_actions.append(r)

        action_map = {}
        sorted_drivers = sorted(self.drivers, key=lambda d: d.money)

        for driver in sorted_drivers:
            if (driver.on_road == 0) & (len(select_actions) != 0):
                random_action = random.choice(select_actions)
                action_map[driver] = random_action
                driver.on_road = 1
                del select_actions[select_actions.index(random_action)]
        r = []
        for driver, request in action_map.items():
            r.append(self.graph.get_edge_data(request.origin, request.destination)["distance"] - self.graph.get_edge_data(driver.pos,
                                                                                                          request.destination)["distance"])
            driver.on_road = 1
            driver.Request = request
        self.time += 1
        if self.time >= 500:
            self.done = True
        return self._state(), r, self.done, {}


'''
class StrictResourceTargetTopEnvironment(TopEnvironment):

    def __init__(self, *args, allow_wait=False, **kvargs):
        super().__init__(*args, allow_wait=allow_wait, **kvargs)
        action_space = len(self.edge_resources)
        if allow_wait:
            action_space += 1
        self._mdp_info.action_space = Discrete(action_space)
        self.edge_ordering = list(self.edge_resources.keys())
        self.shortest_paths = {}
        for edge in self.resource_edges.values():
            self.shortest_paths[edge] = dict(nx.shortest_path(self.graph, target=edge[0], weight='length'))

    def step(self, action):

        if action[0] == len(self.edge_resources):
            return super().step([len(self.actions) - 1])
        edge = self.edge_ordering[action[0]]
        s = None
        rewards = None
        arrived = False
        done = False
        while not arrived and not done:
            path = self.shortest_paths[edge][self.position]
            if len(path) <= 1:
                action = self.edge_indices[edge + (0,)]
                arrived = True
            else:
                next_node = self.shortest_paths[edge][self.position][1]
                action = self.edge_indices[(self.position, next_node, 0)]
            s, r, done, _ = super().step([action])
            self.render(False)
            if rewards is None:
                rewards = r
            else:
                if np.isscalar(r):
                    # rewards += r#每走一步进行累加
                    rewards = r
                else:
                    # rewards[1] += r[1]
                    # rewards[0] += r[0] * self._mdp_info.gamma ** r[1]
                    rewards[1] += r[1]  # 代表travel_time
                    rewards[0][3] += r[0][3] * self._mdp_info.gamma ** r[1]
                    # rewards[0][0]+=r[0][0]
                    for i in range(len(r[0][0])):
                        rewards[0][0][i] += r[0][0][i]
                    rewards[0][1] = r[0][1]
                    rewards[0][2] = r[0][2]
                    if (sum(self.areaInfo.areaCapList) > sum(self.areaInfo.areaVioList)):
                        raise Exception("record error!", rewards)
                    if (sum(r[0][0]) > 0):
                        print(rewards)
            if done:
                break
        return s, rewards, done, {}

'''
