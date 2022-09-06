import random

import numpy as np

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
        self.size = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs, action, reward, next_obs, done):
        # data = (np.array(obs_t, dtype=np.float32), np.array(action, dtype=np.float32),
        #         np.array(reward, dtype=np.float32), np.array(obs_tp1, dtype=np.float32), done)
        data = (obs, action, reward, next_obs, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self.size = min(self.size+1, self._maxsize)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, action, reward, next_obs, done = data
            obses_t.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(next_obs, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def generate_sample_indices(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample(self, batch_size):
        if batch_size > 0:
            idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = range(0, len(self._storage))
        obs_n, acts_n, rew_n, next_obs_n, done_n = self._encode_sample(idx)
        obs_n = np.swapaxes(obs_n, 0, 1)
        acts_n = np.swapaxes(acts_n, 0, 1)
        next_obs_n = np.swapaxes(next_obs_n, 0, 1)
        return obs_n, acts_n, rew_n, next_obs_n, done_n

    def collect(self):
        return self.sample(-1)