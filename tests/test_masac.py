import pytest

import numpy as np

from tf2marl.agents import MASACAgent
from tf2marl.common.util import FakeRun
from tf2marl.common.test_envs.identity_env import IdentityEnv, IdentityEnvBox
from tests.test_agent_env_combination import AgentEnvCombination

import tensorflow as tf

if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],
                                             True)


def test_save_load():
    """
    Tests saving and loading for two agents.
    """
    fp = '/tmp/unittestmaddpg'
    env = IdentityEnv(5, 2)
    agents = [
        MASACAgent(env.observation_space, env.action_space, idx, batch_size=32, buff_size=10000,
                   lr=0.01, num_layer=2,
                   num_units=32, gamma=0.9, tau=0.01, prioritized_replay=True,
                   max_step=5000) for idx in range(2)]

    for idx, agent in enumerate(agents):
        agent.save(fp + str(idx))

    load_agents = [
        MASACAgent(env.observation_space, env.action_space, idx, batch_size=32, buff_size=10000,
                   lr=0.01, num_layer=2,
                   num_units=32, gamma=0.9, tau=0.01, prioritized_replay=True,
                   max_step=5000) for idx in range(2)]

    for idx, agent in enumerate(load_agents):
        agent.load(fp + str(idx))

    def check_for_equal_arrays(list_1, list_2):
        for el1, el2 in zip(list_1, list_2):
            if not (el1 == el2).all():
                return False
        return True

    assert check_for_equal_arrays(load_agents[0].critic_1.model.get_weights(),
                                  agents[0].critic_1.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].critic_1.model.get_weights(),
                                  agents[1].critic_1.model.get_weights())
    assert check_for_equal_arrays(load_agents[0].critic_2.model.get_weights(),
                                  agents[0].critic_2.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].critic_2.model.get_weights(),
                                  agents[1].critic_2.model.get_weights())

    assert check_for_equal_arrays(load_agents[0].critic_1_target.model.get_weights(),
                                  agents[0].critic_1_target.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].critic_1_target.model.get_weights(),
                                  agents[1].critic_1_target.model.get_weights())
    assert check_for_equal_arrays(load_agents[0].critic_2_target.model.get_weights(),
                                  agents[0].critic_2_target.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].critic_2_target.model.get_weights(),
                                  agents[1].critic_2_target.model.get_weights())

    assert check_for_equal_arrays(load_agents[0].v_network.model.get_weights(),
                                  agents[0].v_network.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].v_network.model.get_weights(),
                                  agents[1].v_network.model.get_weights())
    assert check_for_equal_arrays(load_agents[0].v_network_target.model.get_weights(),
                                  agents[0].v_network_target.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].v_network_target.model.get_weights(),
                                  agents[1].v_network_target.model.get_weights())

    assert check_for_equal_arrays(load_agents[0].policy.model.get_weights(),
                                  agents[0].policy.model.get_weights())
    assert check_for_equal_arrays(load_agents[1].policy.model.get_weights(),
                                  agents[1].policy.model.get_weights())


@pytest.mark.slow
@pytest.mark.parametrize("agent_number", [1, 2])
@pytest.mark.parametrize("prioritized_replay", [False])
@pytest.mark.parametrize("entropy_coeff", [0.005, 0.0])
# Soft-Actor Critic does not work well with prioritized replay.
# I assume that his is because both increase the gradient variance
def test_identity_discrete(agent_number, prioritized_replay, entropy_coeff):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    :param model_name: (str) Name of the RL model
    """
    env = IdentityEnv(5, agent_number)

    lr = 0.01 if not prioritized_replay else 0.02
    agents = [
        MASACAgent(env.observation_space, env.action_space, idx, batch_size=32, buff_size=10000,
                   lr=lr, num_layer=2,
                   num_units=32, gamma=0.9, tau=0.01, prioritized_replay=prioritized_replay,
                   max_step=5000, use_gauss_policy=False, use_gumbel=True,
                   entropy_coeff=entropy_coeff,
                   _run=FakeRun())
        for idx in range(agent_number)]

    ag_env_comb = AgentEnvCombination(agents, env)

    target_reward = -1.0 * agent_number
    episode_rewards = ag_env_comb.train(150000, 100,
                                        target_reward)  # starts training after 2000 steps

    assert np.mean(episode_rewards[-10:]) > target_reward

# MASAC is not implemented for continuous environments currently.

# @pytest.mark.slow
# @pytest.mark.parametrize("agent_number", [1, 2])
# @pytest.mark.parametrize("prioritized_replay", [False, True])
# def test_identity_continuous(agent_number, prioritized_replay):
#     """
#     Test if the algorithm (with a given policy)
#     can learn an identity transformation (i.e. return observation as an action), in a continuous environment.
#     :param model_name: (str) Name of the RL model
#     """
#
#     env = IdentityEnvBox(-1, 1, agent_number)
#     lr = 0.01 if not prioritized_replay else 0.02
#     agents = [
#         MASACAgent(env.observation_space, env.action_space, idx, batch_size=100, buff_size=10000,
#                    lr=lr, num_layer=2,
#                    num_units=32, gamma=0.9, tau=0.01, prioritized_replay=prioritized_replay,
#                    max_step=5000, use_gumbel=False, use_gauss_policy=True,
#                    _run=FakeRun()) for idx in range(agent_number)]
#
#     ag_env_comb = AgentEnvCombination(agents, env)
#
#     target_reward = -3.0 * agent_number
#
#     episode_rewards = ag_env_comb.train(6000, 10, target_reward)
#
#     assert np.mean(episode_rewards[-10:]) > target_reward
