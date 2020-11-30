import numpy as np
from tensorflow.keras.utils import to_categorical
from gym import Env
from gym.spaces import Discrete, Box



class IdentityEnv(Env):
    def __init__(self, dim, n, ep_length=100):
        """
        Identity environment for testing purposes, with N agents.
        Observation is for all agents the full state.
        :param dim: (int) the size of the dimensions you want to learn
        :param N: (int) number of agents
        :param ep_length: (int) the length of each episodes in timesteps
        """
        self.action_space = [Discrete(dim) for idx in range(n)]
        self.observation_space = self.action_space
        self.ep_length = ep_length
        self.current_step = 0
        self.dim = dim
        self.n = n
        self.reset()

    def reset(self):
        self.current_step = 0
        self._choose_next_state()
        obs_n = np.array([[self.state] for _ in range(self.n)])
        return obs_n

    def step(self, action_n):
        reward_n = self._get_reward(action_n)
        self._choose_next_state()
        self.current_step += 1
        done_n = [self.current_step >= self.ep_length for _ in range(self.n)]
        obs_n = np.array([[self.state] for _ in range(self.n)])
        return obs_n, reward_n, done_n, {}

    def _choose_next_state(self):
        self.state = to_categorical(self.action_space[0].sample(), num_classes=self.dim)

    def _get_reward(self, action_n):
        reward_n = []
        for ag_idx in range(self.n):
            reward_n.append(-1.0 * np.mean(np.square(self.state - action_n[ag_idx])))
        return np.array(reward_n)

    def render(self, mode='human'):
        pass


class IdentityEnvBox(IdentityEnv):
    def __init__(self, low=-1, high=1, n=2, ep_length=100):
        """
        Identity environment for testing purposes
        :param dim: (int) the size of the dimensions you want to learn
        :param low: (float) the lower bound of the box dim
        :param high: (float) the upper bound of the box dim
        :param n: (int) nubmer of agents
        :param ep_length: (int) the length of each episodes in timesteps
        """
        super(IdentityEnvBox, self).__init__(1, n, ep_length)
        self.action_space = [Box(low=low, high=high, shape=(1,), dtype=np.float32) for idx in range(self.n)]
        self.observation_space = self.action_space
        self.n = n
        self.reset()

    def reset(self):
        self.current_step = 0
        self._choose_next_state()
        obs_n = np.array([[self.state] for _ in range(self.n)])
        return obs_n

    def step(self, action_n):
        reward_n = self._get_reward(action_n)
        self._choose_next_state()
        self.current_step += 1
        done_n = [self.current_step >= self.ep_length for _ in range(self.n)]
        obs_n = np.array([[self.state] for _ in range(self.n)])
        return obs_n, reward_n, done_n, {}

    def _choose_next_state(self):
        self.state = self.observation_space[0].sample()

    def _get_reward(self, action_n):
        reward_n = []
        for ag_idx in range(self.n):
            reward_n.append(-1.0 * np.mean(np.square(self.state - action_n[ag_idx])))
        return np.array(reward_n)


# class IdentityEnvMultiDiscrete(IdentityEnv):
#     def __init__(self, dim, ep_length=100):
#         """
#         Identity environment for testing purposes
#         :param dim: (int) the size of the dimensions you want to learn
#         :param ep_length: (int) the length of each episodes in timesteps
#         """
#         super(IdentityEnvMultiDiscrete, self).__init__(dim, ep_length)
#         self.action_space = MultiDiscrete([dim, dim])
#         self.observation_space = self.action_space
#         self.reset()
