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

            q_next = self._next_q(next_state, absorbing)
            q = rt + self.mdp_info.gamma * q_next

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

            q_next = self._next_q(next_state, absorbing)
            reward = []
            for i in range(len(rt)):
                r = 0
                if len(rt[i]) == 0:
                    reward.append(r)
                else:
                    for j in range(len(rt[i])):
                        r += rt[i][j]
                    reward.append(r)
            print("rt")
            print(reward)
            print("q_next")
            print(q_next)
            print(self.mdp_info.gamma * q_next)
            q = reward + self.mdp_info.gamma * q_next
            td_error = q - self.approximator.predict(state, action)

            self._replay_memory.update(td_error, idxs)

            if approximator is None:
                self.approximator.fit(state, action, q, weights=is_weight,
                                      **self._fit_params)
            else:
                approximator.fit(state, action, q, weights=is_weight,
                                 **self._fit_params)
