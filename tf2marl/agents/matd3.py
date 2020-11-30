import numpy as np
import tensorflow as tf

from gym import Space

from tf2marl.agents.AbstractAgent import AbstractAgent
from tf2marl.agents.maddpg import MADDPGCriticNetwork, MADDPGPolicyNetwork
from tf2marl.common.util import space_n_to_shape_n


class MATD3Agent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 policy_update_freq=2, target_policy_smoothing_eps=0.0, _run=None):
        """
        An object containing critic, actor and training functions for Multi-Agent TD3.
        """
        self._run = _run
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic_1 = MADDPGCriticNetwork(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target = MADDPGCriticNetwork(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target.model.set_weights(self.critic_1.model.get_weights())

        self.critic_2 = MADDPGCriticNetwork(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target = MADDPGCriticNetwork(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target.model.set_weights(self.critic_2.model.get_weights())

        self.policy = MADDPGPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                          self.critic_1, agent_index)
        self.policy_target = MADDPGPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                 self.critic_1, agent_index)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.batch_size = batch_size
        self.decay = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.target_policy_smoothing_eps = target_policy_smoothing_eps
        self.update_counter = 0
        self.agent_index = agent_index

    def action(self, obs):
        """
        Get an action from the non-target policy
        """
        return self.policy.get_action(obs[None])[0]

    def target_action(self, obs):
        """
        Get an action from the non-target policy
        """
        return self.policy_target.get_action(obs)

    def preupdate(self):
        pass

    def update_target_networks(self, tau):
        """
        Implements the updates of the target networks, which slowly follow the real network.
        """
        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights())
            target_net_weights = np.array(target_net.get_weights())
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.critic_1.model, self.critic_1_target.model)
        update_target_network(self.critic_2.model, self.critic_2_target.model)
        update_target_network(self.policy.model, self.policy_target.model)

    def update(self, agents, step):
        """
        Update the agent, by first updating the two critics and then the policy.
        Requires the list of the other agents as input, to determine the target actions.
        """
        assert agents[self.agent_index] is self
        self.update_counter += 1

        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
            self._run.log_scalar('agent_{}.train.mean_weight'.format(self.agent_index), np.mean(weights), step)
            self._run.log_scalar('agent_{}.train.max_weight'.format(self.agent_index), np.max(weights), step)
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)

        # Train the critic, using the target actions in the target critic network, to determine the
        # training target (i.e. target in MSE loss) for the critic update.
        target_act_next = [ag.target_action(obs) for ag, obs in zip(agents, next_obs_n)]
        noise = np.random.normal(0, self.target_policy_smoothing_eps, size=target_act_next[self.agent_index].shape)
        noise = np.clip(noise, -0.5, 0.5)
        target_act_next[self.agent_index] += noise

        critic_outputs = np.empty([2, self.batch_size], dtype=np.float32)  # this is a lot faster than python list plus minimum
        critic_outputs[0] = self.critic_1_target.predict(next_obs_n, target_act_next)[:, 0]
        critic_outputs[1] = self.critic_2_target.predict(next_obs_n, target_act_next)[:, 0]
        target_q_next = np.min(critic_outputs, 0)[:, None]

        q_train_target = rew_n[:, None] + self.decay * target_q_next

        td_loss = np.empty([2, self.batch_size], dtype=np.float32)
        td_loss[0] = self.critic_1.train_step(obs_n, acts_n, q_train_target, weights).numpy()[:, 0]
        td_loss[1] = self.critic_2.train_step(obs_n, acts_n, q_train_target, weights).numpy()[:, 0]
        max_loss = np.max(td_loss, 0)

        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, max_loss + self.prioritized_replay_eps)

        if self.update_counter % self.policy_update_freq == 0:  # delayed policy updates
            # Train the policy.
            policy_loss = self.policy.train(obs_n, acts_n)
            self._run.log_scalar('agent_{}.train.policy_loss'.format(self.agent_index), policy_loss.numpy(), step)
            # Update target networks.
            self.update_target_networks(self.tau)
        else:
            policy_loss = None
        self._run.log_scalar('agent_{}.train.q_loss0'.format(self.agent_index), np.mean(td_loss[0]), step)
        self._run.log_scalar('agent_{}.train.q_loss1'.format(self.agent_index), np.mean(td_loss[1]), step)

        return [td_loss, policy_loss]

    def save(self, fp):
        self.critic_1.model.save_weights(fp + 'critic_1.h5',)
        self.critic_1_target.model.save_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.save_weights(fp + 'critic_2.h5',)
        self.critic_2_target.model.save_weights(fp + 'critic_2_target.h5')

        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        self.critic_1.model.load_weights(fp + 'critic_1.h5',)
        self.critic_1_target.model.load_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.load_weights(fp + 'critic_2.h5',)
        self.critic_2_target.model.load_weights(fp + 'critic_2_target.h5')

        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')
