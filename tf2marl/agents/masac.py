import numpy as np
import tensorflow as tf

from gym import Space

from tf2marl.agents.AbstractAgent import AbstractAgent
from tf2marl.agents.maddpg import MADDPGCriticNetwork, MADDPGPolicyNetwork
from tf2marl.common.util import space_n_to_shape_n, clip_by_local_norm


class MASACAgent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 entropy_coeff=0.2, use_gauss_policy=False, use_gumbel=True, policy_update_freq=1, _run=None,
                 multi_step=1):
        """
        Implementation of Multi-Agent Soft-Actor-Critic, with additional delayed policy updates.
        The implementation here deviates a bit from the standard soft actor critic, by not using the
        value function and target value function, but instead using 2 q functions with 2 targets each.
        Using the value function could also be tested.

        Also the learning of the entropy temperature could still be implemented. Right now setting the entropy
        coefficient is very important.
        todo: entropy temperature learning
        todo: gaussian policy
        note: does not use value function but only two q functions
        note: ensure gumbel softmax entropy is calculated correctly
        """
        self._run = _run
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic_1 = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_1_target.model.set_weights(self.critic_1.model.get_weights())
        self.critic_2 = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target = MADDPGCriticNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)
        self.critic_2_target.model.set_weights(self.critic_2.model.get_weights())

        # this was proposed to be used in the original SAC paper but later they got rid of it again
        self.v_network = ValueFunctionNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)  # unused
        self.v_network_target = ValueFunctionNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index)  # unused
        self.v_network_target.model.set_weights(self.v_network.model.get_weights())  # unused

        self.policy = MASACPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1, entropy_coeff,
                                         agent_index, self.critic_1, use_gauss_policy, use_gumbel, prioritized_replay_eps)
        self.policy_target = MASACPolicyNetwork(num_layer, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1, entropy_coeff,
                                         agent_index, self.critic_1, use_gauss_policy, use_gumbel, prioritized_replay_eps)
        self.policy_target.model.set_weights(self.policy.model.get_weights())


        self.use_gauss_policy = use_gauss_policy
        self.use_gumbel = use_gumbel
        self.policy_update_freq = policy_update_freq

        self.batch_size = batch_size
        self.decay = gamma
        self.tau = tau
        self.entropy_coeff = entropy_coeff
        self.update_counter = 0
        self.agent_index = agent_index
        self.multi_step = multi_step

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

        update_target_network(self.v_network.model, self.v_network_target.model)
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
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)

        # Train the critic, using the target actions in the target critic network, to determine the
        # training target (i.e. target in MSE loss) for the critic update.
        next_act_sampled_n = [ag.target_action(next_obs) for ag, next_obs in zip(agents, next_obs_n)]
        if self.use_gauss_policy:
            logact_probs = self.policy.action_logprob(next_obs_n[self.agent_index], next_act_sampled_n[self.agent_index])[:, None]  # only our own entropy is 'controllable'
            entropy = -logact_probs
        elif self.use_gumbel:
            action_probs = self.policy.get_all_action_probs(next_obs_n[self.agent_index])
            action_log_probs = np.log(action_probs + self.prioritized_replay_eps)
            buff = -action_probs * action_log_probs
            entropy = np.sum(buff, 1)

        critic_outputs = np.empty([2, self.batch_size], dtype=np.float32)  # this is a lot faster than python list plus minimum
        critic_outputs[0] = self.critic_1_target.predict(next_obs_n, next_act_sampled_n)[:, 0]
        critic_outputs[1] = self.critic_2_target.predict(next_obs_n, next_act_sampled_n)[:, 0]
        q_min = np.min(critic_outputs, 0)[:, None]

        target_q = rew_n[:, None] + self.decay * (q_min + self.entropy_coeff * entropy)

        #### Separate Value Function version ####
        # target_q = rew_n[:, None] + self.decay * self.v_network_target.predict(next_obs_n)
        # # sac does this "cross" updating between Q and V functions
        #
        # critic_outputs = np.empty([2, self.batch_size], dtype=np.float32)  # this is a lot faster than python list plus minimum
        # critic_outputs[0] = self.critic_1.predict(obs_n, act_sampled_n)[:, 0]
        # critic_outputs[1] = self.critic_2.predict(obs_n, act_sampled_n)[:, 0]
        # q_min = np.min(critic_outputs, 0)[:, None]
        # target_v = q_min + self.entropy_coeff * entropy

        td_loss = np.empty([2, self.batch_size], dtype=np.float32)
        td_loss[0] = self.critic_1.train_step(obs_n, acts_n, target_q, weights).numpy()[:, 0]
        td_loss[1] = self.critic_2.train_step(obs_n, acts_n, target_q, weights).numpy()[:, 0]
        # v_loss = self.v_network.train_step(obs_n, target_v, weights).numpy()[:, 0]
        td_loss_max = np.max(td_loss, 0)

        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_loss_max + self.prioritized_replay_eps)

        # Train the policy.
        if self.update_counter % self.policy_update_freq == 0:  # delayed policy updates
            policy_loss = self.policy.train(obs_n, acts_n)
            # Update target networks.
            self.update_target_networks(self.tau)
            self._run.log_scalar('agent_{}.train.policy_loss'.format(self.agent_index), policy_loss.numpy(), step)
        else:
            policy_loss = None

        self._run.log_scalar('agent_{}.train.q_loss0'.format(self.agent_index), np.mean(td_loss[0]), step)
        self._run.log_scalar('agent_{}.train.q_loss1'.format(self.agent_index), np.mean(td_loss[1]), step)
        self._run.log_scalar('agent_{}.train.entropy'.format(self.agent_index), np.mean(entropy), step)

        return [td_loss, policy_loss]

    def save(self, fp):
        self.critic_1.model.save_weights(fp + 'critic_1.h5',)
        self.critic_2.model.save_weights(fp + 'critic_2.h5')
        self.critic_1_target.model.save_weights(fp + 'critic_target_1.h5',)
        self.critic_2_target.model.save_weights(fp + 'critic_target_2.h5')

        self.v_network.model.save_weights(fp + 'value.h5')
        self.v_network_target.model.save_weights(fp + 'value_target.h5')
        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')


    def load(self, fp):
        self.critic_1.model.load_weights(fp + 'critic_1.h5')
        self.critic_2.model.load_weights(fp + 'critic_2.h5')
        self.critic_1_target.model.load_weights(fp + 'critic_target_1.h5',)
        self.critic_2_target.model.load_weights(fp + 'critic_target_2.h5')

        self.v_network.model.load_weights(fp + 'value.h5')
        self.v_network_target.model.load_weights(fp + 'value_target.h5')
        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')


class ValueFunctionNetwork(MADDPGCriticNetwork):
    def __init__(self, num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n, act_type,
                 agent_index):
        """
        Implementation of a critic to represent the Value function. Almost the same
        as the critic network in MADDPG, but only uses one input.
        """
        super().__init__(num_hidden_layers, units_per_layer, lr, obs_n_shape,
                         act_shape_n, act_type, agent_index)

        # replace network structure
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(num_hidden_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}value_out'.format(agent_index))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n) if len(self.obs_shape_n) > 1 else self.obs_input_n[0]
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n, outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n):
        """
        Predict the value of the given state.
        """
        return self._predict_internal(obs_n)

    @tf.function
    def train_step(self, obs_n, target, weights):
        """
        Train the value function estimator, for one gradient step. With clipped gradients.
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer(obs_n) if len(obs_n) > 1 else obs_n[0]
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            v_pred = self.output_layer(x)
            td_loss = tf.math.square(target - v_pred)
            loss = tf.reduce_mean(td_loss * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss

class MASACPolicyNetwork(MADDPGPolicyNetwork):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, entropy_coeff, agent_index, q_network, use_gaussian, use_gumbel,
                 numeric_eps):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final
        layer. Currently only implemented for discrete spaces with a gumbel policy.
        """
        super().__init__(num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                         gumbel_temperature, q_network, agent_index)
        self.use_gaussian = not self.use_gumbel
        self.entropy_coeff = entropy_coeff
        self.numeric_eps = numeric_eps

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])

        self.hidden_layers = []
        for idx in range(num_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        if self.use_gumbel:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='linear',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))
        else:
            self.output_layer = tf.keras.layers.Dense(self.act_shape, activation='tanh',
                                                      name='ag{}pol_out{}'.format(agent_index, idx))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    @tf.function
    def get_all_action_probs(self, obs):
        """
        Return the action probabilities. Only works for discrete action spaces.
        """
        logits = self.forward_pass(obs)
        return tf.math.softmax(logits)

    @classmethod
    def gaussian_sample(cls, logits):
        return

    @classmethod
    def gaussian_prob(cls, logits, actions):
        return

    @tf.function
    def action_logprob(self, obs, action):
        """
        Returns the log of the probability of the action, given the state.
        Can be used for vectors.
        :param state:
        :param action:
        :return:
        """
        logits = self.forward_pass(obs)

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            x = obs_n[self.agent_index]
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            x = self.output_layer(x)
            act_n = tf.unstack(act_n)
            if self.use_gumbel:
                logits = x  # log probabilities of the gumbel softmax dist are the output of the network
                act_n[self.agent_index] = self.gumbel_softmax_sample(logits)
                act_probs = tf.math.softmax(logits)
                entropy = - tf.math.reduce_sum(act_probs * tf.math.log(act_probs + self.numeric_eps), 1)
            elif self.use_gaussian:
                logits = x
                act_n[self.agent_index] = self.gaussian_sample(logits)
                entropy = - self.action_logprob(obs_n[self.agent_index], act_n[self.agent_index])
            q_value = self.q_network._predict_internal(obs_n + act_n)

            loss = -tf.math.reduce_mean(q_value + self.entropy_coeff * entropy)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss
