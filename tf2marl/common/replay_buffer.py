"""
This file implements the replay buffer. It is based on the replay buffer from
stable-baselines, with some changes.
"""
import random

import numpy as np

from tf2marl.common.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        # data = (np.array(obs_t, dtype=np.float32), np.array(action, dtype=np.float32),
        #         np.array(reward, dtype=np.float32), np.array(obs_tp1, dtype=np.float32), done)
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def generate_sample_indices(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.generate_sample_indices(batch_size)
        else:
            idxes = range(0, len(self._storage))
        obs_n, acts_n, rew_n, next_obs_n, done_n = self._encode_sample(idxes)
        obs_n = np.swapaxes(obs_n, 0, 1)
        acts_n = np.swapaxes(acts_n, 0, 1)
        next_obs_n = np.swapaxes(next_obs_n, 0, 1)
        return obs_n, acts_n, rew_n, next_obs_n, done_n

    def collect(self):
        return self.sample(-1)


class EfficientReplayBuffer(object):
    def __init__(self, size, n_agents, obs_shape_n, act_shape_n):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._obs_n = []
        self._acts_n = []
        self._obs_tp1_n = []
        self._n_agents = n_agents
        for idx in range(n_agents):
            self._obs_n.append(np.empty([size, obs_shape_n[idx, 0]], dtype=np.float32))
            self._acts_n.append(np.empty([size, act_shape_n[idx][0]], dtype=np.float32))
            self._obs_tp1_n.append(np.empty([size, obs_shape_n[idx,0]], dtype=np.float32))
        self._done = np.empty([size], dtype=np.float32)
        self._reward = np.empty([size], dtype=np.float32)
        self._maxsize = size
        self._next_idx = 0
        self.full = False
        self.len = 0

    def __len__(self):
        return self.len

    def add(self, obs_t, action, reward, obs_tp1, done):
        for ag_idx in range(self._n_agents):
            self._obs_n[ag_idx][self._next_idx] = obs_t[ag_idx]
            self._acts_n[ag_idx][self._next_idx]  = action[ag_idx]
            self._obs_tp1_n[ag_idx][self._next_idx] = obs_tp1[ag_idx]
        self._reward[self._next_idx] = reward
        self._done[self._next_idx] = done

        if not self.full:
            self._next_idx = self._next_idx + 1
            if self._next_idx > self._maxsize - 1:
                self.full = True
                self.len = self._maxsize
                self._next_idx = self._next_idx % self._maxsize
            else:
                self.len = self._next_idx - 1
        else:
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > self.len:
            raise RuntimeError('Too few samples in buffer to generate batch.')

        indices = np.random.randint(self.len, size=[batch_size])

        obs_n = []
        acts_n = []
        next_obs_n = []
        for ag_idx in range(self._n_agents):
            obs_n.append(self._obs_n[ag_idx][indices])
            acts_n.append(self._acts_n[ag_idx][indices].copy())
            next_obs_n.append(self._obs_tp1_n[ag_idx][indices])

        rew = self._reward[indices]
        done = self._done[indices]
        return obs_n, acts_n, rew, next_obs_n, done


class PrioritizedReplayBuffer(EfficientReplayBuffer):
    def __init__(self, size, n_agents, obs_shape_n, act_shape_n, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size, n_agents, obs_shape_n, act_shape_n)
        self.alpha = alpha
        for idx in range(n_agents):
            self._obs_n.append(np.empty([size, obs_shape_n[idx, 0]], dtype=np.float32))
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_n, action_n, reward, obs_next_n, done):
        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).add(obs_n, action_n, reward,
                                                 obs_next_n, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_indices_proportional(self, batch_size):
        indices = []
        for batch_idx in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            indices.append(idx)
        return indices

    def sample(self, batch_size, beta=0):
        indices = self._sample_indices_proportional(batch_size)
        weights = []
        priority_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (priority_min * len(self)) ** (-beta)
        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)

        obs_n = []
        acts_n = []
        next_obs_n = []
        for ag_idx in range(self._n_agents):
            obs_n.append(self._obs_n[ag_idx][indices])
            acts_n.append(self._acts_n[ag_idx][indices].copy())
            next_obs_n.append(self._obs_tp1_n[ag_idx][indices])

        rew = self._reward[indices]
        done = self._done[indices]
        return obs_n, acts_n, rew, next_obs_n, done, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param indices: ([int]) List of indices of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
