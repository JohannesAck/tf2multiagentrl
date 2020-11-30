from abc import ABC, abstractmethod
from builtins import NotImplementedError

from tf2marl.common.replay_buffer import EfficientReplayBuffer, PrioritizedReplayBuffer
from tf2marl.common.util import LinearSchedule

class AbstractAgent(ABC):
    def __init__(self, buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay=False,
                 alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6):
        self.batch_size = batch_size
        self.prioritized_replay_eps = prioritized_replay_eps
        self.act_shape_n = act_shape_n
        self.obs_shape_n = obs_shape_n
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(int(buff_size), len(obs_shape_n),
                                                         obs_shape_n, act_shape_n, alpha)
            assert max_step is not None
            self.beta_schedule = LinearSchedule(max_step, 1.0, initial_beta)
        else:
            self.replay_buffer = EfficientReplayBuffer(int(buff_size), len(obs_shape_n),
                                                       obs_shape_n, act_shape_n)
        self.prioritized_replay = prioritized_replay

    @abstractmethod
    def action(self, obs):
        """
        Get an action from the non-target policy
        :param obs: A batch of Observations, ndarray
        :return:  A batch of actions, tf.Tensor
        """
        pass

    @abstractmethod
    def target_action(self, obs):
        """
        Get an action from the target policy
        :param obs: A batch of Observations, ndarray
        :return:  A batch of actions, tf.Tensor
        """

        pass

    def add_transition(self, obs_n, act_n, rew, new_obs_n, done_n):
        """
        Adds a transition to the replay buffer.
        """
        self.replay_buffer.add(obs_n, act_n, rew, new_obs_n, float(done_n))

    @abstractmethod
    def update(self, agents, step):
        """
        Updates the Agent according to its implementation, i.e. performs (a) learning step.
        :return: Q_loss, Policy_loss
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, fp):
        """
        Saves the Agent to the specified directory.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, fp):
        """
        Loads weights from a given file.
        Has to be called on a fitting agent, that was created with the same hyperparameters.
        """
        raise NotImplementedError()
