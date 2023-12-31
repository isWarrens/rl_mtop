import datetime
import pathlib
import random
from time import time
import json
import os

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN, DoubleDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import ExponentialParameter, LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory

from environment import TopEnvironment
from smdp_dqn import SMDPDQN

from area import AreaInfo
from preProcess.utils import change_node_to_int


class MyCore(Core):
    def _step(self, render):
        # 每次选动作之前把司机状态更新
        # 更新每个司机的状态
        for driver in self.mdp.drivers:
            if driver.on_road == 1:
                driver.start_time += self.mdp.timestep
                if (self.mdp.graph.get_edge_data(driver.Request.origin, driver.Request.destination)["distance"] -
                    self.mdp.graph.get_edge_data(driver.pos,
                                                 driver.Request.origin)[
                        "distance"]) / driver.speed <= driver.start_time:
                    driver.on_road = 0
                    driver.pos = driver.Request.destination
                    driver.money += self.mdp.graph.get_edge_data(driver.Request.origin,
                                                                 driver.Request.destination)["distance"]
        actions = []
        for idx in range(len(self.mdp.drivers)):
            action = self.agent.draw_action(self._state)
            actions.append(action[0])
        # 选出来的点
        next_state, reward, absorbing, _ = self.mdp.step(actions)

        self._episode_steps += 1

        last = not (
                self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        self._state = next_state.copy()

        return state, actions, reward, next_state, absorbing, last


class ResourceObservation:
    """
    This class parses the state to to an observation processable for the neural network.
    """

    def init(self, env):
        self.shape = (len(env.drivers), 4)

    def __init__(self, use_weekdays=False, distance_normalization=3000):
        self.shape = 4
        self.model = None
        self.shortest_paths_lengths = None
        self.distance_normalization = distance_normalization
        self.use_weekdays = use_weekdays

    def __len__(self):
        return np.prod(self.shape)

    def __call__(self, env, *args, **kwargs):
        state = np.zeros(self.shape)

        for i, driver in enumerate(env.drivers):
            '''
                1.State[i,0]:free状态是1 占用是0
                4.State[i,1]:graph的所有node。
                5.State[i,2]:时间戳
                6.state[i,3] 每个司机的收益
            '''
            state[i, 0] = 1 if driver.on_road == 0 else 0
            state[i, 1] = env.drivers[i].pos if env.drivers[i].on_road == 0 else -1
            state[i, 2] = env.time
            state[i, 3] = env.drivers[i].money

        return state


class ResourceObservation2:
    """
    This class parses the state  to an observation processable for the neural network.
    This implementation computes the time of arrival at each resource and checks if the resource will be in violation
    at arrival time.
    """

    def __init__(self, speed, use_weekdays=False, distance_normalization=3000):
        self.shape = None
        self.model = None
        self.speed = speed
        self.shortest_paths_lengths = None
        self.distance_normalization = distance_normalization
        self.use_weekdays = use_weekdays

    def init(self, env):
        self.shortest_paths_lengths = {}
        for edge in env.resource_edges.values():
            self.shortest_paths_lengths[edge] = dict(
                nx.shortest_path_length(env.graph, target=edge[0], weight='length'))
        '''
        求解从源节点到目标节点的最短路径。
        Return type:	list
        '''
        self.shape = (len(env.devices), 4 + 1 + 1 + 1 + 1 + 1 + (7 if self.use_weekdays else 0))

    def __len__(self):
        return np.prod(self.shape)

    def __call__(self, env, *args, **kwargs):
        if self.shape is None:
            self.init(env)

        state = np.zeros(self.shape)
        # print(env.time)
        dt = datetime.datetime.fromtimestamp(env.time)
        weekday = dt.weekday()
        t = (dt - datetime.datetime.combine(dt.date(), datetime.time())).seconds
        # 当前时刻与一天开始时刻之间的差值转换到秒
        t = (t - env.start_hour * 60 * 60) / ((env.end_hour - env.start_hour) * (60 * 60))
        # 从开始时刻到现在的时间

        for i, device in enumerate(env.device_ordering):
            distance = self.shortest_paths_lengths[env.resource_edges[device]][env.position]
            walk_time = distance / self.speed

            device_state = env.spot_states[device]
            '''
            env.spot_states：一个dict，关于device_id和当前状态
            self.FREE: "blue",
            self.OCCUPIED: 'green',
            self.VIOLATION: 'red',
            self.VISITED: 'yellow'
            FREE = 0
            OCCUPIED = 1
            VIOLATION = 2
            VISITED = 3
            '''
            if device_state == TopEnvironment.OCCUPIED and \
                    env.time + walk_time > env.potential_violation_times[device]:
                device_state = TopEnvironment.VIOLATION
            state[i, device_state] = 1
            if self.use_weekdays:
                state[i, 4 + weekday] = 1

            state[i, -5] = walk_time / 3600.
            state[i, -4] = t
            state[i, -3] = t + walk_time / ((env.end_hour - env.start_hour) * (60 * 60))
            state[i, -2] = min(2,
                               (env.time + walk_time - env.potential_violation_times[device]) / env.allowed_durations[
                                   device]) \
                if device in env.potential_violation_times and env.spot_states[device] != TopEnvironment.VISITED else -1
            '''
            env.time:当前时间
            env.potential_violation_times:关于device_id和arrval+duration的dictionary，正处于violation的device
            env.allowed_durations[device]:关于device_id和duration的dictionary
            '''
            state[i, -1] = distance / self.distance_normalization

        return state


'''
他这里的edge_ordering是
'''


class GraphConvolutionResourceNetwork(nn.Module):
    n_features = 32
    n_scaling_features = 64

    def init(self, env):
        self.shape = (len(env.graph.drivers), 4)
        self.actions_num = len(env.graph.nodes)

    def __init__(self, input_shape, output_shape,
                 graph=None, resource_embeddings=0, nn_scaling=False, nodes_num=0, driver_nums=0,
                 **kwargs):
        super().__init__()

        self.shape = None
        n_output = output_shape[-1]
        self.nn_scaling = nn_scaling
        self.actions_num = nodes_num
        print(self.actions_num)

        distances = torch.zeros(graph.number_of_nodes(), graph.number_of_nodes())
        distances = distances / distances.max()
        self.register_buffer("distances", distances)
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                if nx.shortest_path_length(graph, node1, node2) == np.nan:
                    print("valid" + node1, node2)
                self.distances[node1][node2] = nx.shortest_path_length(graph, node1, node2)
        has_nan = torch.isnan(self.distances).any()

        if nn_scaling:
            self.scaling = torch.nn.Sequential(
                torch.nn.Linear(1, self.n_scaling_features),
                torch.nn.Sigmoid(),
                torch.nn.Linear(self.n_scaling_features, driver_nums),
                torch.nn.Sigmoid()
            )
        else:
            self.scaling = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
            self.scaling.data.uniform_()

        self.resource_embeddings = torch.nn.Parameter(data=torch.Tensor(driver_nums, resource_embeddings),
                                                      requires_grad=True)
        self.resource_embeddings.data.uniform_()
        self.init_encoding = torch.nn.Sequential(
            torch.nn.Linear(input_shape[-1] + resource_embeddings, self.n_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            torch.nn.Linear(self.n_features, 1)
        )

    def forward(self, state, action=None):
        res_enc = self.resource_embeddings.repeat(state.shape[0], 1, 1)
        res_enc = torch.cat([state.float(), res_enc], dim=2)
        res_enc = self.init_encoding(res_enc)

        if self.nn_scaling:
            A = self.scaling(self.distances.unsqueeze(-1))
        else:
            A = torch.exp(-self.scaling * self.distances)
        A = A.mean(dim=1)
        # A = A / A.sum(1).max()
        x = torch.matmul(A, res_enc)

        q = self.model(x).squeeze(-1)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


class SimpleResourceNetwork(nn.Module):
    """
    Neural network implementation.
    """

    n_features = 128

    def __init__(self, input_shape, output_shape, edges=None, resources=None, load_path=None, **kwargs):
        super().__init__()

        n_output = output_shape[-1]
        self.n_resources = len(resources)
        self.n_edges = len(edges)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input_shape), self.n_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_features, self.n_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_features, n_output)
        )

        if load_path is not None:
            self.state_dict(torch.load(load_path))

    def forward(self, state, action=None):
        # inp = torch.zeros((state.shape[0], 4 * self.n_resources + 1), dtype=torch.float)
        # r = torch.arange(state.shape[0]).unsqueeze(1).repeat(1,self.n_resources)
        # inp[r, 4*torch.arange(self.n_resources) + state[:, -self.n_resources:].long()] = 1
        # inp[:, -1] = state[:, self.n_edges]
        q = self.model(state.view(state.shape[0], -1).float())

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def compute_J(dataset, gamma=1.):
    rewardList = list()
    utilityList = list()
    ulity = 0
    for i in range(len(dataset)):
        print(dataset[i][0])
        x = dataset[i][2]
        if len(x) != 0:
            rewardList.append(x)
            for j in range(len(x)):
                ulity += x[j]

    utilityList.append(ulity)
    state = dataset[len(dataset) - 1]
    arr = []
    for p in range(len(state[0])):
        arr.append(state[0][p][3])
    arr = np.array(arr)
    fairness = np.std(arr)
    return rewardList, utilityList, fairness


def experiment(mdp, params, prob=None):
    # Argument parser
    # parser = argparse.ArgumentParser()
    #
    # args = parser.parse_args()

    scores = list()

    optimizer = dict()
    if params['optimizer'] == 'adam':
        optimizer['class'] = optim.Adam
        optimizer['params'] = dict(lr=params['learning_rate'])
    elif params['optimizer'] == 'adadelta':
        optimizer['class'] = optim.Adadelta
        optimizer['params'] = dict(lr=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        # 默认值
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=params['learning_rate'],
                                   alpha=params['decay'])
        '''
        'learning_rate': 0.0012,
        'lr_decay': 1.,
        '''
    elif params['optimizer'] == 'rmspropcentered':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=params['learning_rate'],
                                   alpha=params['decay'],
                                   centered=True)
    else:
        raise ValueError

    # DQN learning run

    # Summary folder
    folder_name = os.path.join(PROJECT_DIR, 'logs', params['name'])
    if params['save']:
        pathlib.Path(folder_name).mkdir(parents=True)

    # Policy
    epsilon = ExponentialParameter(value=params['initial_exploration_rate'],
                                   exp=params['exploration_rate'],
                                   min_value=params['final_exploration_rate'],
                                   size=(1,))
    '''
    参数根据使用的次数变化
    'exploration_rate': .1,
    'final_exploration_rate': .01,
    'final_exploration_frame': 200000,
    '''
    epsilon_random = Parameter(value=1)
    epsilon_test = Parameter(value=0.01)
    pi = EpsGreedy(epsilon=epsilon_random)

    # ϵ ( < 1 ) 的概率随机选择未知的一个动作，剩下1 − ϵ 的概率选择已有动过中动作价值最大的动作

    # Approximator
    input_shape = mdp.observation.shape

    N = {
        'SimpleResourceNetwork': SimpleResourceNetwork,
        'GraphConvolutionResourceNetwork': GraphConvolutionResourceNetwork,
    }[params['network']]

    N.n_features = params['hidden']

    approximator_params = dict(
        network=N,
        input_shape=input_shape,
        graph=mdp.graph,
        allow_wait=params['allow_wait'],
        long_term_q=params['long_term_q'],
        resource_embeddings=params['resource_embeddings'],
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        nodes_num=mdp.info.action_space.n,
        driver_nums=params['driver_nums'],
        n_features=params['hidden'],
        # 256
        optimizer=optimizer,
        loss=F.smooth_l1_loss,
        nn_scaling=params['nn_scaling'],
        # quiet=False,
        use_cuda=params['cuda'],
        load_path=params.get('load_path', None)
    )

    approximator = TorchApproximator

    replay_memory = PrioritizedReplayMemory(
        params['initial_replay_size'], params['max_replay_size'], alpha=.6,
        beta=LinearParameter(.4, threshold_value=1,
                             n=params['max_steps'] // params['train_frequency'])
    )

    # Agent
    algorithm_params = dict(
        batch_size=params['batch_size'],
        # 128
        n_approximators=1,
        target_update_frequency=params['target_update_frequency'] // params['train_frequency'],
        replay_memory=replay_memory,
        initial_replay_size=params['initial_replay_size'],
        max_replay_size=params['max_replay_size']
    )

    clz = DoubleDQN if mdp.info.gamma >= 1 else SMDPDQN
    print("gamma: " + str(mdp.info.gamma))
    # clz=DoubleDQN
    agent = clz(mdp.info, pi, approximator,
                approximator_params=approximator_params,
                **algorithm_params)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(agent.approximator._impl.model._optimizer,
                                                   step_size=1,
                                                   gamma=params['lr_decay'],
                                                   last_epoch=-1)  # params['max_steps'] // params['train_frequency']
    # Algorithm
    core = MyCore(agent, mdp)

    if 'weights' in params:
        best_weights = np.load(params['weights'])
        agent.approximator.set_weights(best_weights)
        agent.target_approximator.set_weights(best_weights)
    else:
        best_weights = agent.approximator.get_weights()

    # RUN
    pi.set_epsilon(epsilon_test)
    eval_days = [39]
    ds = core.evaluate(initial_states=eval_days, render=params['save'], quiet=tuning)
    # print("dataset:")
    # print(ds)

    rewardList, ulity, fairness = compute_J(ds, params['gamma'])
    print(ulity)
    print(rewardList)
    print(fairness)

    # Fill replay memory with random dataset
    print_epoch(0)
    start = time()
    core.learn(n_steps=params['initial_replay_size'],
               n_steps_per_fit=params['initial_replay_size'], quiet=tuning)

    runtime = time() - start
    steps = 0

    if params['save']:
        with open(folder_name + "/params.json", "w") as f:
            json.dump(params, f, indent=4)
        if isinstance(agent, DQN):
            np.save(folder_name + '/weights-exp-0-0.npy',
                    agent.approximator.get_weights())

    if params['save']:
        np.save(folder_name + '/scores.npy', scores)
    for n_epoch in range(1, int(params['max_steps'] // params['evaluation_frequency'] + 1)):
        print_epoch(n_epoch)
        print('- Learning:')
        # learning step
        pi.set_epsilon(epsilon)
        # mdp.set_episode_end(True)
        start = time()
        core.learn(n_steps=params['evaluation_frequency'],
                   n_steps_per_fit=params['train_frequency'], quiet=tuning)
        runtime += time() - start
        steps += params['evaluation_frequency']
        lr_scheduler.step()

        if params['save']:
            if isinstance(agent, DQN):
                np.save(folder_name + '/weights-exp-0-' + str(n_epoch) + '.npy',
                        agent.approximator.get_weights())

        print('- Evaluation:')
        # evaluation step
        pi.set_epsilon(epsilon_test)
        ds = core.evaluate(initial_states=eval_days, render=params['save'], quiet=tuning)
        rewardList, ulity, fairness = compute_J(ds, params['gamma'])
        print(ulity)
        print(rewardList)
        print(fairness)

    print('---------- FINAL EVALUATION ---------')
    agent.approximator.set_weights(best_weights)
    agent.target_approximator.set_weights(best_weights)
    pi.set_epsilon(epsilon_test)
    eval_days = [39]
    ds = core.evaluate(initial_states=eval_days, render=params['save'], quiet=tuning)
    rewardList, ulity, fairness = compute_J(ds, params['gamma'])
    print("---ulity------")
    print(ulity)
    print("----fairness---")
    print(fairness)

    return scores


def train_top(external_params=None):
    params = {
        'seed': 352625,
        # Downtown
        # 'area': ['Courtney', 'Family', 'Victoria Market', 'Regency', 'West Melbourne', 'Princes Theatre', 'Titles',
        #          'Magistrates', 'Supreme', 'Mint', 'Library',
        #          'Hyatt', 'Banks', 'Twin Towers', 'Spencer', 'McKillop', 'Tavistock',
        #          'City Square', 'Chinatown', 'RACV', 'Markilles', 'Degraves'],
        'area': 'Queensberry',
        # 'area': 'Docklands',
        'network': 'GraphConvolutionResourceNetwork',
        'gamma_half_time': 3600.,
        'start_hour': 7,
        'end_hour': 19,
        'initial_replay_size': 5000,
        'max_replay_size': 100000,
        'optimizer': 'rmsprop',
        'learning_rate': 0.0012,
        'lr_decay': 1.,
        'hidden': 256,
        'decay': .99,
        'batch_size': 128,
        'target_update_frequency': 50000,
        'evaluation_frequency': 50000,
        'average_updates': 8,
        'max_steps': 1000000,
        'initial_exploration_rate': 1.,
        'exploration_rate': .1,
        'final_exploration_rate': .01,
        'final_exploration_frame': 200000,
        'allow_wait': False,
        'long_term_q': False,
        'resource_embeddings': 0,
        'nn_scaling': True,
        'afterstates': True,
        'use_weekdays': False,
        'save': True,
        'cuda': torch.cuda.is_available(),
        'name': time_str,
        'driver_nums': 5,
        # 一个step跑多少时间
        'final_time': 5000,
        'fairness_discount': 0.4,
        'driver_speed': 50
    }

    if external_params is not None:
        params.update(external_params)
    params['train_frequency'] = np.ceil(params['batch_size'] // params['average_updates'])  # np.ceil大于等于的最小整数.
    if 'gamma_half_time' in params:
        params['gamma'] = np.power(0.5, 1. / params['gamma_half_time'])

    if 'network_int' in params:
        params['network'] = ['GraphConvolutionResourceNetwork', 'AttentionResourceNetwork'][params['network_int']]
    params['name'] = params.get('name', "") + "_" + \
                     ("" if external_params is None or 'seed' not in external_params else str(external_params['seed']))

    params['resource_embeddings'] = int(params['resource_embeddings'])
    print("name", params['name'])

    np.random.seed(params['seed'])
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    observation = ResourceObservation(use_weekdays=params['use_weekdays'])
    mdp = TopEnvironment(params['gamma'], params['driver_nums'], params['driver_speed'], observation, 0, 1,
                         params['final_time'])
    observation.init(mdp)
    experiment(mdp, params, prob=None)

    mdp.close()


if __name__ == '__main__':
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    PROJECT_DIR = os.getcwd()

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tuning = False
    # train_top({
    #     "seed": 87532,
    #     "network": "GraphConvolutionResourceNetwork",
    #     "gamma_half_time": 1800,
    #     "start_hour": 7,
    #     "end_hour": 19,
    #     "initial_replay_size": 5000,
    #     "max_replay_size": 125000,
    #     "optimizer": "rmsprop",
    #     "learning_rate": 0.001,
    #     "lr_decay": 1.0,
    #     "hidden": 256,
    #     "decay": 0.99,
    #     "batch_size": 128,
    #     "target_update_frequency": 50000,
    #     "evaluation_frequency": 25000,
    #     "average_updates": 16,
    #     "max_steps": 1000000,
    #     "initial_exploration_rate": 1.0,
    #     "exploration_rate": 1e-05,
    #     "final_exploration_rate": 0.1,
    #     "final_exploration_frame": 200000,
    #     "allow_wait": False,
    #     "long_term_q": False,
    #     "resource_embeddings": 0,
    #     "nn_scaling": True,
    #     "afterstates": True,
    #     "use_weekdays": False,
    #     "train_frequency": 8.0,
    #     "weights": 'weights-exp-0-16.npy'
    # })

    train_top({
        'seed': 352625,
        'area': ['Docklands', 'Southbank', 'Queensberry'],
        # 'area': 'Princes Theatre',
        'network': 'GraphConvolutionResourceNetwork',
        # 'network': 'Network',
        # 'gamma': 1,

        'gamma_half_time': 1800.,
        # 'gamma': np.power(0.5, 1./3600.),
        'start_hour': 7,
        'end_hour': 19,
        'initial_replay_size': 5000,
        'max_replay_size': 100000,
        'optimizer': 'rmsprop',
        'learning_rate': 0.0012,
        'lr_decay': 1.,
        'hidden': 256,
        'decay': .99,
        'batch_size': 128,
        'target_update_frequency': 50000,
        'evaluation_frequency': 100,
        'average_updates': 8,
        'max_steps': 2000,
        'initial_exploration_rate': 1.,
        'exploration_rate': .1,
        'final_exploration_rate': .01,
        'final_exploration_frame': 200000,
        'allow_wait': False,
        'long_term_q': False,
        'resource_embeddings': 0,
        'nn_scaling': True,
        'afterstates': True,
        'use_weekdays': False,
        'save': True,
        'cuda': False,
        'name': time_str,
        'driver_nums': 10,
        # 一个step跑多少时间
        'final_time': 50,
        'driver_speed': 90
    })
