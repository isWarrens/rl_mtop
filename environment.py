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

class TopEnvironment(Environment):
    """
    Implementation of the travelling officer environment.
    """

    FREE = 0
    OCCUPIED = 1
    VIOLATION = 2
    VISITED = 3
    LOCATION_TABLE = "sensors"
    FULL_JOIN = " " + LOCATION_TABLE + " join restrictions on bay_id=BayID join events on restrictions.DeviceID=events.DeviceId "

    def __init__(self, database, area, observation, delta_degree=0.1, speed=5., gamma=1, start_hour=8, end_hour=22, days_loaded=1, train_days=None, add_time=False, allow_wait=False, project_dir=None):
        # self.area = area
        self.next_event = None
        self.time = None
        self.position = None
        self.done = False
        self.speed = speed
        self.departures = None
        self.spot_states = None
        self.violation_times = None
        self.potential_violation_times = None
        self.allowed_durations = None
        self.delayed_arrival = None
        self.observation = observation
        self.render_images = None
        self.position_plots = None
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.add_time = add_time
        self.days_loaded = days_loaded
        self.days_consumed = np.inf
        self.day = None
        self.end_day = None
        self.areaInfo=None

        # self.train_days = train_days if train_days is not None else list(range(1, 356))
        self.train_days=[39]

        conn = sqlite3.connect('file:%s?mode=ro' % database, uri=True)
        cursor = conn.cursor()
        if area is None:
            self.areas = []
        elif isinstance(area, list) or isinstance(area, tuple):
            self.areas = area
        else:
            self.areas = [area]

        self.areaInfo=AreaInfo(self.areas)

        if len(self.areas) == 0:
            where_areas = " 1 = 1 "  # if no area specified, use all!
        else:
            where_areas = " or ".join(map(lambda a: " Area=\"" + a + "\" ", self.areas))


        cursor.execute("SELECT min(lat) as south, max(lat) as north, min(lon) as west, max(lon) as east "
                       "from devices where " + where_areas)
        #筛选指定area的数据

        row = cursor.fetchone()

        sqlx = 'select DeviceID, ArrivalTime, DepartureTime, duration*60, Area ' \
               'from events join durations on durations.sign=events.sign ' \
               'where ' + where_areas + ' and "Vehicle Present"="True" order by ArrivalTime'
        self.all_events = pd.read_sql(sqlx, conn, coerce_float=True, params=None)


        self.events = None
        self.event_idx = None

        dx = max(0.01, row[1] - row[0]) * delta_degree
        dy = max(0.01, row[3] - row[2]) * delta_degree
        self.north = row[1] + dx
        self.south = row[0] - dx
        self.west = row[2] - dy
        self.east = row[3] + dy

        # 生成的自己地图
        if project_dir is None:
            project_dir = os.getcwd()
        data_dir = os.path.join(project_dir, "data")
        area_hash = hashlib.md5(json.dumps(sorted(area)).encode("utf8")).hexdigest()

        graph_file = 'graph_' + area_hash + '.gml'
        try:
            self.graph = ox.load_graphml(graph_file, folder=data_dir)
        except FileNotFoundError:
            self.graph = ox.graph_from_bbox(
                south=self.south, north=self.north, west=self.west, east=self.east, network_type='walk')
            #创建graph
            # ox.plot_graph(self.graph)
            ox.save_graphml(self.graph, filename=graph_file, folder=data_dir)
        print("graph nodes: ", len(self.graph.nodes), ", edges: ", len(self.graph.edges))
        self.actions = tuple(self.graph.edges)

        if allow_wait:
            self.actions = self.actions + ('wait', )
        self.edge_indices = {edge: i for i, edge in enumerate(self.actions)}
        #关于edge和它的index的dictionary
        self.edge_lengths = nx.get_edge_attributes(self.graph, 'length')

        self.edge_resources = dict()#edge-deviceid
        self.resource_edges = dict()#deviceid-edge
        self.devices = dict()
        self.device_ordering = []
        self.spot_plots = dict()

        devices_file = os.path.join(data_dir, "devices_" + area_hash + ".pckl")
        try:
            with open(devices_file, "rb") as f:
                self.devices, self.device_ordering, self.edge_resources, self.resource_edges = pickle.load(f)
        except:
            print("cannot load devices file", devices_file)
            fig, ax = ox.plot_graph(self.graph, show=False, close=False)
            res = cursor.execute("SELECT lat, lon, DeviceID " +
                                 "from devices " +
                                 "where " + where_areas)
            for lat, lon, device_id, *vio_times in tqdm(res.fetchall()):
                # g = ox.truncate_graph_bbox(self.graph, lat+dy, lat-dy, lon+dx, lon-dx, True, True)
                g = ox.truncate_graph_dist(self.graph, ox.get_nearest_node(self.graph, (lat, lon)), 500, retain_all=True)#删除距离超过500的点
                nearest = ox.get_nearest_edge(g, (lat, lon))#查找距离最近的边
                #寻找距离最近的edge
                edge = (nearest[1], nearest[2])
                self.device_ordering.append(device_id)
                #添加device id
                self.devices[device_id] = (lat, lon, edge)
                if edge in self.edge_resources:
                    edge_resources = self.edge_resources[edge]
                else:
                    edge_resources = []#用于记录device_id
                    self.edge_resources[edge] = edge_resources
                edge_resources.append(device_id)
                self.resource_edges[device_id] = edge
                ax.plot(*nearest[0].xy, color='blue')
                ax.scatter(lon, lat, s=2, c="red")

            with open(devices_file, "wb") as f:
                pickle.dump((self.devices, self.device_ordering, self.edge_resources, self.resource_edges), f)
            plt.show()

        print(len(self.devices), "parking devices")

        mdp_info = MDPInfo(Discrete(2), Discrete(len(self.actions)), gamma, np.inf)
        super().__init__(mdp_info)

        cursor.close()
        conn.close()

    # 生成地图，我们只用一个地图数据，因不需要此增量
    def create_graph(self):
        # 读取CSV文件
        data = pd.read_csv('bay_locations.csv')

        # 根据CSV文件中的经纬度计算区域范围
        south = data['lat'].min()
        north = data['lat'].max()
        west = data['lon'].min()
        east = data['lon'].max()

        # 从OpenStreetMap获取地图数据
        graph = ox.graph_from_bbox(south=south, north=north, west=west, east=east, network_type='walk')

        # 打印图形节点和边的数量
        print("graph nodes: ", len(graph.nodes), ", edges: ", len(graph.edges))

        # 可选：绘制地图
        ox.plot_graph(graph)

        # 可选：保存地图为图形文件
        ox.save_graphml(graph, filename='graph.gml')
        self.graph = graph

    def close(self):
        pass

    def reset(self, state=None):



        year = 2017
        self.done = False
        self.position = min(list(self.graph.nodes))

        if self.days_consumed < self.days_loaded:
            # self.time = int(datetime.strptime(("%03d %d %02d:00" % (self.day + self.days_consumed, year, self.start_hour)), "%j %Y %H:%M").strftime("%s"))
            # self.end_day = int(datetime.strptime(("%03d %d %02d:00" % (self.day + self.days_consumed, year, self.end_hour)), "%j %Y %H:%M").strftime("%s"))
            self.time = int(
                datetime.strptime(("%03d %d %02d:00" % (self.day + self.days_consumed, year, self.start_hour)),
                                  "%j %Y %H:%M").timestamp())
            self.end_day = int(
                datetime.strptime(("%03d %d %02d:00" % (self.day + self.days_consumed, year, self.end_hour)),
                                  "%j %Y %H:%M").timestamp())
            self._update_spots()
            return self._state()

        self.areaInfo.initial()

        self.delayed_arrival = []
        self.departures = []
        self.violation_times = []
        self.potential_violation_times = dict()
        self.allowed_durations = dict()
        self.spot_states = {id: self.FREE for id in self.devices}
        self.render_images = []
        # print(state)
        if state is None:
            self.day = np.random.choice(self.train_days)
            print("self.day:" + str(self.day))
        else:
            self.day = int(state)
            print("self.day:"+str(self.day))
            # self.day=39
        # start = datetime.strptime(("%03d %d %02d:00" % (self.day, year, self.start_hour)), "%j %Y %H:%M").strftime("%s")
        # end = datetime.strptime(("%03d %d %02d:00" % (self.day + (self.days_loaded-1), year, self.end_hour)), "%j %Y %H:%M").strftime("%s")
        # self.end_day = int(datetime.strptime(("%03d %d %02d:00" % (self.day, year, self.end_hour)), "%j %Y %H:%M").strftime("%s"))
        # print(self.day)
        start ="%d"% (datetime.strptime(("%03d %d %02d:00" % (self.day, year, self.start_hour)), "%j %Y %H:%M").timestamp())
        end = "%d"% (datetime.strptime(("%03d %d %02d:00" % (self.day + (self.days_loaded - 1), year, self.end_hour)),
                                "%j %Y %H:%M").timestamp())
        self.end_day = int((datetime.strptime("%03d %d %02d:00" % (self.day, year, self.end_hour), "%j %Y %H:%M").timestamp()))
        # print("start"+str(start))
        # print("end"+str(end))

        self.events = self.all_events.loc[
            (self.all_events['DepartureTime'] > int(start))
            & (self.all_events['ArrivalTime'] < int(end))
        ]

        print(self.events)
        self.events.to_csv(".//fig//On-street_Car_Parking_Sensor_Data_-_2017_02_08.csv")

        self.areaInfo.device2Area(self.events)
        print(self.areaInfo.deviceAreaDict)
        self.event_idx = 0

        self.days_consumed = 0
        self._next_event()
        self.time = int(start)
        self._update_spots()
        print("self.spot_states")
        print(self.spot_states)
        return self._state()

    def _next_event(self):
        while True:
            if self.event_idx + 1 >= len(self.events):
                self.next_event = None
                break

            self.event_idx += 1
            self.next_event = self.events.iloc[self.event_idx]


            if self.next_event[0] in self.devices:
                break

    def _state(self):
        return self.observation(self)

    def _update_spots(self):


        if self.time > self.end_day:
            self.done = True
            self.days_consumed += 1
            return

        while True:
            # print("next_event")
            # print(self.next_event)

            times = [
                nsmallest(1, self.violation_times)[0][0] if len(self.violation_times) > 0 else np.inf,
                nsmallest(1, self.departures)[0][0] if len(self.departures) > 0 else np.inf,
                nsmallest(1, self.delayed_arrival)[0][0] if len(self.delayed_arrival) > 0 else np.inf,
                self.next_event['ArrivalTime'],
            ]

            if min(times) > self.time:
                break
            arg = np.argmin(times)
            # print("times")
            # print(times)
            # print("arg"+str(arg))

        # while self.next_event['ArrivalTime'] <= self.time:
            if arg == 2 or arg == 3:
                if arg == 3:
                    #如果下个事件的arrival time是最小值，针对下个事件进行运算，压入对应的堆中
                    device_id, arrival, departure, duration, area = self.next_event
                    #DeviceID, ArrivalTime, DepartureTime, duration*60
                else:  # if arg == 2:
                    t, device_id, arrival, departure, duration ,area = heappop(self.delayed_arrival)
                if departure >= self.time:

                    #如果当前还未离开或还未到达
                    if self.spot_states[device_id] != self.FREE:
                        #如果已经到了
                        t = max([t for t, d in self.departures if d == device_id])
                        #departures堆里面选取时间的最大值
                        if t+1 > departure:
                            #如果departure小于departures堆中的最大值，压入delayed_arrival堆中
                            heappush(self.delayed_arrival, (t+1, device_id, arrival, departure, duration))
                    else:
                        #如果
                        self.spot_states[device_id] = self.OCCUPIED
                        heappush(self.departures, (departure, device_id))
                        assert sum([d == device_id for t, d in self.departures]) == 1
                        self.potential_violation_times[device_id] = arrival + duration
                        self.allowed_durations[device_id] = duration
                        if departure > arrival + duration:
                            heappush(self.violation_times, (arrival + duration, device_id))
                self._next_event()

                if self.next_event is None or self.next_event['ArrivalTime'] >= self.end_day:
                    self.done = True
                    self.days_consumed += 1
                    break
        # while len(self.violation_times) > 0 and nsmallest(1, self.violation_times)[0][0] <= self.time:
            if arg == 0:
                t, device_id = heappop(self.violation_times)
                self.spot_states[device_id] = self.VIOLATION
                self.areaInfo.recordAreaOfDeviceVio(device_id)
                assert device_id in self.potential_violation_times
                # self.areaInfo.recordAreaOfDeviceVio(device_id)


        # while len(self.departures) > 0 and nsmallest(1, self.departures)[0][0] <= self.time:
            if arg == 1:
                t, device_id = heappop(self.departures)
                self.spot_states[device_id] = self.FREE
                if device_id in self.potential_violation_times:
                    del self.potential_violation_times[device_id]
                    del self.allowed_durations[device_id]
                    assert sum([d == device_id for t, d in self.violation_times]) == 0
            pass

    def legal_actions(self, position=None):
        if position is None:
            position = self.position
        return [self.edge_indices[e + (0,)] for e in self.graph.out_edges(position)]

    def step(self, action):
        self.areaInfo.initialStep()
        edge = self.actions[action[0]]

        reward = []
        if edge == 'wait':
            travel_time = 10
            # reward += 0.0005
        else:
            if edge[0] != self.position:
                return self._state(), -100, self.done, {}

            self.position = edge[1]
            #更新agent位置
            travel_time = self.edge_lengths[edge] / self.speed

        self.time += travel_time
        self._update_spots()

        if edge != 'wait':
            edge = (edge[0], edge[1])
           # reward+=1
            if edge in self.edge_resources:
                # print("edde in resource")
                for device_id in self.edge_resources[edge]:
                    if self.spot_states[device_id] == self.VIOLATION:
                        self.spot_states[device_id] = self.VISITED
                        self.areaInfo.recordAreaOfDeviceCap(device_id)

                #单做一个rewardlist，分别记录所有area的capture。更新Q-TABLE使用综合减去标准差。multi-objective RL
        rewardValue=self.areaInfo.caculStepReward()
        reward=[self.areaInfo.areaCapStepList,self.areaInfo.areaCapList,self.areaInfo.areaVioList,rewardValue]




        if self.add_time:
            r = np.array((reward, travel_time),dtype=object)
        else:
            r = reward

        return self._state(), r, self.done, {}


    def _update_plot(self, plot_entry):
        color_map = {
            self.FREE: "blue",
            self.OCCUPIED: 'green',
            self.VIOLATION: 'red',
            self.VISITED: 'yellow'
        }
        position, spot_states = plot_entry
        for device_id, path in self.spot_plots.items():
            path.set_edgecolor(color_map[spot_states[device_id]])
        features = self.graph.nodes[position]
        self.position_plots.set_offsets([[features['x'], features['y']]])

    def render(self, show=False):
        plot_entry = (self.position, copy.deepcopy(self.spot_states))
        self.render_images.append(plot_entry)
        if show:
            self._update_plot(plot_entry)
            plt.draw()
            plt.pause(0.1)

    def save_rendered(self, file, max_frames=500):
        fig, ax = ox.plot_graph(self.graph, show=False, close=False, node_size=0)

        for id, value in self.devices.items():
            lat, lon, edge = value
            self.spot_plots[id] = ax.scatter(lon, lat, s=2, c="red", zorder=11)
        self.position_plots = ax.scatter(0, 0, s=50, c='blue', zorder=10)

        frames = min(max_frames, len(self.render_images))
        pbar = tqdm(total=frames)

        def update(i):
            self._update_plot(self.render_images[i])
            pbar.update(1)

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)

        ani.save(file)

        plt.close(fig)
        pbar.close()


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
            return super().step([len(self.actions)-1])
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
                    #rewards += r#每走一步进行累加
                    rewards=r
                else:
                    # rewards[1] += r[1]
                    # rewards[0] += r[0] * self._mdp_info.gamma ** r[1]
                    rewards[1] += r[1]#代表travel_time
                    rewards[0][3] += r[0][3]*self._mdp_info.gamma**r[1]
                    # rewards[0][0]+=r[0][0]
                    for i in range(len(r[0][0])):
                        rewards[0][0][i]+=r[0][0][i]
                    rewards[0][1]=r[0][1]
                    rewards[0][2]=r[0][2]
                    if (sum(self.areaInfo.areaCapList) > sum(self.areaInfo.areaVioList)):
                        raise Exception("record error!", rewards)
                    if(sum(r[0][0])>0):
                        print(rewards)
            if done:
                break
        return s, rewards, done, {}

if __name__ == "__main__":
    # 读取CSV文件
    project_dir = os.getcwd()
    data_dir = os.path.join(project_dir, "data")
    data = pd.read_csv(data_dir + '/bay_sensors_vio_loc_03_19.csv')

    # 根据CSV文件中的经纬度计算区域范围
    south = data['lat'].min()
    north = data['lat'].max()
    west = data['lon'].min()
    east = data['lon'].max()

    # 从OpenStreetMap获取地图数据
    graph = ox.graph_from_bbox(south=south, north=north, west=west, east=east, network_type='walk')

    # 打印图形节点和边的数量
    print("graph nodes: ", len(graph.nodes), ", edges: ", len(graph.edges))

    # 可选：绘制地图
    ox.plot_graph(graph)

