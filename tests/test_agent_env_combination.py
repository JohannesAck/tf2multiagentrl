"""
Implements a combined env with agents and environment, to be used in unittests.
"""

import numpy as np

class AgentEnvCombination(object):
    def __init__(self, agents, env):
        """
        This object implements a simple training loop given an agent and an environment.
        It is only used in the unit-tests and therefore has very reduced logging.
        """
        self.agents = agents
        self.env = env
        self.n = len(agents)

    def train(self, n_steps, update_rate, target_reward):
        """
        Train agent for set number of steps.
        :param n_steps:
        :param update_rate:
        :return:
        """
        episode_rewards = [0]
        agent_rewards = [[0] for _ in range(self.n)]
        obs_n = self.env.reset()

        for train_step in range(n_steps):
            # get action
            action_n = [agent.action(obs.astype(np.float32))
                        for agent, obs in zip(self.agents, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = self.env.step(action_n)
            done = any(done_n)

            # collect experience
            for idx, agent in enumerate(self.agents):
                agent.add_transition(obs_n, np.array(action_n), rew_n[idx], new_obs_n, done)
            obs_n = new_obs_n

            for idx, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[idx][-1] += rew

            if done:
                print('{}/{}: ep_reward = {}'.format(train_step, n_steps, episode_rewards[-1]))
                obs_n = self.env.reset()
                episode_rewards.append(0)  # add element for next episode
                for a in agent_rewards:
                    a.append(0)
                if np.mean(episode_rewards[-15:]) > target_reward:
                    return episode_rewards

            for agent in self.agents:
                if len(agent.replay_buffer) > agent.batch_size * 20:  # replay buffer is not large enough
                    if train_step % update_rate == 0:  # only update every 100 steps
                        q_loss, pol_loss = agent.update(self.agents, train_step)

        return episode_rewards