import numpy as np
from gym.spaces import Box, Discrete, Tuple
import tensorflow as tf
from tf2marl.multiagent.multi_discrete import MultiDiscrete

def space_n_to_shape_n(space_n):
    """
    Takes a list of gym spaces and returns a list of their shapes
    """
    return np.array([space_to_shape(space) for space in space_n])

def space_to_shape(space):
    """
    Takes a gym.space and returns its shape
    """
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [space.n]
    elif isinstance(space, MultiDiscrete):
        return [sum(space.high - space.low + 1)]
    elif isinstance(space, Tuple):
        # Assuming each element in the tuple is a Discrete space
        # Modify this logic if your Tuple spaces have different structures
        return [sum(element.n for element in space.spaces)]
    else:
        raise RuntimeError(f"Unknown space type: {type(space)}. Can't return shape.")


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    FROM STABLE BASELINES
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.current_step = 0
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def clip_by_local_norm(gradients, norm):
    """
    Clips gradients by their own norm, NOT by the global norm
    as it should be done (according to TF documentation).
    This here is the way MADDPG does it.
    """
    for idx, grad in enumerate(gradients):
        gradients[idx] = tf.clip_by_norm(grad, norm)
    return gradients


class FakeRun(object):
    def __init__(self):
        """
        A fake run object as sacred uses, meant to be used as a replacement in unit test.
        """
        self.counter = 0

    def log_scalar(self, name, val, step):
        self.counter += 1


def softmax_to_argmax(action_n, agents):
    """
    If given a list of action probabilities performs argmax on each of them and outputs
    a one-hot-encoded representation.
    Example:
        [0.1, 0.8, 0.1, 0.0] -> [0.0, 1.0, 0.0, 0.0]
    :param action_n: list of actions per agent
    :param agents: list of agents
    :return List of one-hot-encoded argmax per agent
    """
    hard_action_n = []
    for ag_idx, (action, agent) in enumerate(zip(action_n, agents)):
        hard_action_n.append(tf.keras.utils.to_categorical(np.argmax(action), agent.act_shape_n[ag_idx,0]))

    return hard_action_n
