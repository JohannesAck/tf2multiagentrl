import numpy as np
import tensorflow as tf
from gym import Space
from gym.spaces import Discrete

from tf2marl.agents.AbstractAgent import AbstractAgent
from tf2marl.common.util import space_n_to_shape_n, clip_by_local_norm


class MAD3PGAgent(AbstractAgent):
    def __init__(self, obs_space_n, act_space_n, agent_index, batch_size, buff_size, lr, num_layer, num_units, gamma,
                 tau, prioritized_replay=False, alpha=0.6, max_step=None, initial_beta=0.6, prioritized_replay_eps=1e-6,
                 _run=None, num_atoms=51, min_val=-150, max_val=0):
        """
        An object containing critic, actor and training functions.
        :param num_layer:
        """
        self._run = _run

        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = space_n_to_shape_n(obs_space_n)
        act_shape_n = space_n_to_shape_n(act_space_n)
        super().__init__(buff_size, obs_shape_n, act_shape_n, batch_size, prioritized_replay, alpha, max_step, initial_beta,
                         prioritized_replay_eps=prioritized_replay_eps)

        act_type = type(act_space_n[0])
        self.critic = CatDistCritic(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index, num_atoms, min_val, max_val)
        self.critic_target = CatDistCritic(2, num_units, lr, obs_shape_n, act_shape_n, act_type, agent_index, num_atoms, min_val, max_val)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        self.policy = MAD3PGPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                          self.critic, agent_index)
        self.policy_target = MAD3PGPolicyNetwork(2, num_units, lr, obs_shape_n, act_shape_n[agent_index], act_type, 1,
                                                 self.critic, agent_index)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.batch_size = batch_size
        self.agent_index = agent_index
        self.decay = gamma
        self.tau = tau

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

        update_target_network(self.critic.model, self.critic_target.model)
        update_target_network(self.policy.model, self.policy_target.model)

    def update(self, agents, step):
        """
        Update the agent, by first updating the critic and then the actor.
        Requires the list of the other agents as input, to determine the target actions.
        """
        assert agents[self.agent_index] is self

        if self.prioritized_replay:
            obs_n, acts_n, rew_n, next_obs_n, done_n, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step))
        else:
            obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)
            weights = tf.ones(rew_n.shape)

        # Train the distributional critic, this is a little bit confusing, but described well in Bellemare ICML17
        target_act_next = [a.target_action(obs) for a, obs in zip(agents, next_obs_n)]
        target_prob_next = self.critic_target.predict_probs(next_obs_n, target_act_next)
        q_next = np.sum(target_prob_next * self.critic.atoms, 1)  # note: maybe change this to tf to speed_up

        atoms_next = rew_n[:, None] + self.decay * self.critic.atoms
        atoms_next = np.clip(atoms_next, self.critic.min_val, self.critic.max_val).astype(np.float32)

        target_prob = self.project_distribution(atoms_next, target_prob_next)

        # apply update
        td_loss = self.critic.train_step(obs_n, acts_n, target_prob, weights).numpy()

        # Train the policy.
        policy_loss = self.policy.train(obs_n, acts_n)

        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_loss + self.prioritized_replay_eps)

        # Update target networks.
        self.update_target_networks(self.tau)

        self._run.log_scalar('agent_{}.train.E_q_mean'.format(self.agent_index), np.mean(q_next), step)
        self._run.log_scalar('agent_{}.train.policy_loss'.format(self.agent_index), policy_loss.numpy(), step)
        self._run.log_scalar('agent_{}.train.q_loss0'.format(self.agent_index), np.mean(td_loss), step)

        return [td_loss, policy_loss]

    def project_distribution(self, atoms_next, target_prob_next):
        """
        Projects the distribution onto the new support.
        Includes in the comments a non-vectorized version, which is a lot slower, although the new one is still a bit
        slow.
        TODO: this numpy only solution is a bit slow.
        """
        b = (atoms_next - self.critic.min_val) / self.critic.delta_atom  # b = continuous 'index' of atom being projected to
        lower = np.floor(b)
        upper = np.ceil(b)
        # determine/interpolate membership to different atoms
        density_lower = target_prob_next * (upper + np.float32(
            lower == upper) - b)  # note: not sure about lower == upper, that's stolen from ShangtongZhang
        density_upper = target_prob_next * (b - lower)
        # sum up membergship from all projected atoms to target probability
        target_prob = np.zeros(target_prob_next.shape, dtype=np.float32)
        for batch_idx in range(self.batch_size):  # todo it would be better to vectorize per atom, not per batch idx...
            target_prob[batch_idx, np.int32(lower[batch_idx, :])] += density_lower[batch_idx, :]
            target_prob[batch_idx, np.int32(upper[batch_idx, :])] += density_upper[batch_idx, :]
        return target_prob


    @tf.function
    def project_distribution_tf(self, atoms_next, target_prob_next):
        """
        Projects the distribution onto the new support.
        Includes in the comments a non-vectorized version, which is a lot slower, although the new one is still a bit
        slow.
        DOES NOT WORK, THIS IS DIFFICULT TO IMPLEMENT BECAUSE YOU CAN NOT ASSIGN STRIDES IN TF!
        THERE IS A TF1.X SOLUTION HERE:
        https://github.com/google/dopamine/blob/70a6cb03ce70ae7369d64768e8922bb20ede6a27/dopamine/agents/rainbow/rainbow_agent.py#L329
        """
        target_prob = tf.zeros([self.batch_size, self.critic.n_atoms], tf.float32)
        for batch_idx in range(self.batch_size):
            b = (atoms_next[batch_idx] - self.critic.min_val) / self.critic.delta_atom  # b = continuous 'index'/position of atom being projected to
            lower = tf.math.floor(b)
            upper = tf.math.ceil(b)
            # determine/interpolate membership to different atoms
            density_lower = target_prob_next * (upper + tf.cast(lower == upper, tf.float32) - b)
            # note: not sure about lower == upper, that's stolen from ShangtongZhang
            density_upper = target_prob_next * (b - lower)
            lower = tf.cast(lower, tf.int32)
            upper = tf.cast(upper, tf.int32)
            target_prob[batch_idx, lower] += density_lower
            target_prob[batch_idx, upper] += density_upper
        return target_prob

    def save(self, fp):
        self.critic.model.save_weights(fp + 'critic.h5',)
        self.critic_target.model.save_weights(fp + 'critic_target.h5')
        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        self.critic.model.load_weights(fp + 'critic.h5')
        self.critic_target.model.load_weights(fp + 'critic_target.h5')
        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')

class CatDistCritic(object):
    def __init__(self, num_hidden_layers, units_per_layer, lr, obs_n_shape, act_shape_n, act_type, agent_index,
                 n_atoms, min_val, max_val):
        """
        Implementation of a critic that outputs a categorical distribution, similar to how it was used in both
        D4PG and the original Bellemare Distributional paper.
        regression ANN.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n
        self.act_type = act_type
        self.n_atoms = n_atoms
        self.atoms = np.linspace(min_val, max_val, n_atoms)
        self.delta_atom = (max_val - min_val) / (n_atoms - 1)
        self.min_val = min_val
        self.max_val = max_val

        self.clip_norm = 0.5
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(num_hidden_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(self.n_atoms, activation='softmax',
                                                  name='ag{}crit_out{}'.format(agent_index, idx))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n,  # list concatenation
                                    outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict_probs(self, obs_n, act_n):
        """
        Return the probabilities for the given input.
        """
        return self._predict_internal(obs_n + act_n)

    def predict_expectation(self, obs_n, act_n):
        """
        Predict the expectation of the distribution for given input.
        """
        probs = self._predict_internal(obs_n + act_n)
        dist = probs * self.atoms
        return tf.math.reduce_sum(dist, 1)


    @tf.function
    def _predict_internal(self, concatenated_input):
        """
        Internal function, because concatenation can not be done in tf.function
        """
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        # x = tf.math.softmax(x)
        return x

    def train_step(self, obs_n, act_n, target_prob, weights):
        """
        Train the critic network with the observations, actions, rewards and next observations, and next actions.
        """
        return self._train_step_internal(obs_n + act_n, target_prob, weights)

    @tf.function
    def _train_step_internal(self, concatenated_input, target_prob, weights):
        """
        Internal function, because concatenation can not be done inside tf.function
        """
        with tf.GradientTape(persistent=True) as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            x = self.output_layer(x)
            # q_pred = tf.math.softmax(x)
            q_pred = x

            crossent_loss = tf.losses.binary_crossentropy(target_prob, q_pred)
            # crossent_loss = -tf.math.reduce_sum(tf.math.log(target_prob + 1e-16) * q_pred, 1)
            #loss = tf.reduce_mean(crossent_loss * weights)
            loss = crossent_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)

        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return crossent_loss


class MAD3PGPolicyNetwork(object):
    def __init__(self, num_layers, units_per_layer, lr, obs_n_shape, act_shape, act_type,
                 gumbel_temperature, q_network, agent_index):
        """
        Implementation of the policy network, with optional gumbel softmax activation at the final layer.
        """
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.act_type = act_type
        if act_type is Discrete:
            self.use_gumbel = True
        else:
            self.use_gumbel = False
        self.gumbel_temperature = gumbel_temperature
        self.q_network = q_network
        self.agent_index = agent_index
        self.clip_norm = 0.5

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

    @classmethod
    def gumbel_softmax_sample(cls, logits):
        """
        Produces Gumbel softmax samples from the input log-probabilities (logits).
        These are used, because they are differentiable approximations of the distribution of an argmax.
        """
        uniform_noise = tf.random.uniform(tf.shape(logits))
        gumbel = -tf.math.log(-tf.math.log(uniform_noise))
        noisy_logits = gumbel + logits  # / temperature
        return tf.math.softmax(noisy_logits)

    @tf.function
    def get_action(self, obs):
        x = obs
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # log probabilities of the gumbel softmax dist are the output of the network

        if self.use_gumbel:
            samples = self.gumbel_softmax_sample(outputs)
            return samples
        else:
            return outputs

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
                # act_n = tf.stack(act_n)
            else:
                act_n[self.agent_index] = x
            q_value = self.q_network.predict_expectation(obs_n, act_n)
            policy_regularization = tf.math.reduce_mean(tf.math.square(x))
            loss = -tf.math.reduce_mean(q_value) + 1e-3 * policy_regularization  # gradient plus regularization

        gradients = tape.gradient(loss, self.model.trainable_variables)  # todo not sure if this really works
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
