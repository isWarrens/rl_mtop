import numpy as np
from mushroom_rl.algorithms.value import DoubleDQN
import statistics
from area import AreaInfo

class SMDPDQN(DoubleDQN):

    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)

    def _fit_standard(self, dataset, approximator=None):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, rt, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            # reward, time = rt[:, 0], rt[:, 1]
            rewardList,time=rt[:,0],rt[:,1]
            time=np.array(time,dtype=np.float64)
            # if self._clip_reward:
            #     # reward = np.clip(reward, -1, 1)
            #     rewardList = np.clip(rewardList,-1,1)
            reward = []
            for areaReward in rewardList:
                # areaRewardVal=0
                # for i in range(len(areaReward[0])):
                #     areaRewardVal+=areaReward[0][i]
                # areaRewardVal-=AreaInfo.areaCapStdTool(areaReward)
                reward.append(areaReward[3])
            reward=np.array(reward)
            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma ** time * q_next

            if approximator is None:
                self.approximator.fit(state, action, q, **self._fit_params)
            else:
                approximator.fit(state, action, q, **self._fit_params)

    def _fit_prioritized(self, dataset, approximator=None):
        self._replay_memory.add(
            dataset, np.ones(len(dataset)) * self._replay_memory.max_priority)
        if self._replay_memory.initialized:
            state, action, rt, next_state, absorbing, _, idxs, is_weight = \
                self._replay_memory.get(self._batch_size)

            # reward, time = rt[:, 0], rt[:, 1]
            rewardList, time = rt[:, 0], rt[:, 1]
            time = np.array(time, dtype=np.float64)
            # print("rt")
            # print(rt)
            # if self._clip_reward:
            #     # reward = np.clip(reward, -1, 1)
            #     rewardList = np.clip(rewardList, -1, 1)
            reward = []
            for areaReward in rewardList:
                reward.append(areaReward[3])
            reward = np.array(reward)
            q_next = self._next_q(next_state, absorbing)
            # print("time")
            # print(time)
            # print("q_next")
            # print(q_next)
            # print(self.mdp_info.gamma ** time * q_next)

            q = reward + self.mdp_info.gamma ** time * q_next
            td_error = q - self.approximator.predict(state, action)

            self._replay_memory.update(td_error, idxs)

            if approximator is None:
                self.approximator.fit(state, action, q, weights=is_weight,
                                      **self._fit_params)
            else:
                approximator.fit(state, action, q, weights=is_weight,
                                 **self._fit_params)
