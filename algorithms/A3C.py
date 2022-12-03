import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Permute
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from multiprocessing import cpu_count
import threading
from threading import Thread

from racetrack_env import RaceTrackEnv

####### HYPERPARAMETERS #######
NUM_ACTIONS = 1
FC_LAYERS = 2
FC_WIDTH = 256
OBS_DIM = (2,18,18)
NUM_WORKERS = None


LR = 5e-4
NUM_EPISODES = 0
UPDATE_GLOBAL_FREQ = 500
SAVE_MODEL = False
MIN_REWARD = 0
EXP_DIR = ""
LOG_FREQ = 100
EVAL_FREQ = 1000
RMSPROP_EPSILON = 0.01

A3C_GAMMA = 1
ACTOR_COEF = 1.0
CRITIC_COEF = 0.5
ENTROPY_COEF = 0.01
LR_DECAY = True
###############################

class ActorCritic(Model):
    def __init__(self):
        super().__init__('ActorCriticModel')
        self.create_model()
        self.num_actions = NUM_ACTIONS
        self.fc_layers = FC_LAYERS
        self.fc_width = FC_WIDTH
        self.obs_dim = OBS_DIM

    def create_model(self):
        self.feature_extractor = Sequential() ### LINEAR MODEL
        self.feature_extractor.add(Permute((3,2,1), input_shape=self.obs_dim))
        self.feature_extractor.add(Flatten())

        # Take output feature dimensions
        feature_output_dim = self.feature_extractor.layers[-1].output_shape

        self.actor_network = Sequential()
        self.var_network = Sequential()
        self.critic_network = Sequential()

        for _ in range(self.fc_layers):
            self.actor_network.add(Dense(self.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.var_network.add(Dense(self.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.critic_network.add(Dense(self.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
        
        self.actor_network.add(Dense(self.num_actions, activation='tanh', kernel_initializer = 'glorot_uniform'))
        self.var_network.add(Dense(self.num_actions, activation='softmax', kernel_initializer = 'glorot_uniform'))
        self.critic_network.add(Dense(1, kernel_initializer = 'glorot_uniform'))

        self.actor_network.build(feature_output_dim)
        self.var_network.build(feature_output_dim)
        self.critic_network.build(feature_output_dim)

    def call(self, inputs):
        feats = self.feature_extractor(inputs)
        action_output = self.actor_network(feats)
        var_output = self.var_network(feats)
        value_output = self.critic_network(feats)
        return action_output, var_output, tf.squeeze(value_output)
    
    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)

        feats = self.feature_extractor(obs)
        mu = self.actor_network(feats)
        var = self.var_network(feats)

        probability_density = tfp.distributions.Normal(mu, var)
        action = probability_density.sample(self.num_actions)
        action = tf.clip_by_value(action, -1, 1)

        return action.numpy()


class A3CAgent():
    def __init__(self):
        self.policy_global = ActorCritic()   # Initialize Global Network
        self.num_workers = cpu_count() if NUM_WORKERS == None else NUM_WORKERS

    def learn(self, env, opt):
        workers = []

        for i in range(self.num_workers):
            env = RaceTrackEnv(opt)
            workers.append(A3C_Worker(env, self.policy_global, i))
    
        for worker in workers: worker.start()    
        for worker in workers: workers.join()

class A3C_Worker(Thread):
    
    global_best = 0
    global_episode = 0
    global_total_reward = []
    global_total_loss = []
    global_actor_loss = []
    global_critic_loss = []
    global_entropy_loss = []
    save_lock = threading.Lock()

    def __init__(self, env, policy_global, worker_idx):
        Thread.__init__(self)
        self.env = env
        self.policy_global = policy_global
        self.num_episodes = NUM_EPISODES
        self.num_actions = NUM_ACTIONS
        self.obs_dim = OBS_DIM
        self.update_global_freq = UPDATE_GLOBAL_FREQ
        self.lr = LR

        self.worker_idx = worker_idx
        self.save_model = SAVE_MODEL
        self.min_reward = MIN_REWARD
        self.exp_dir = EXP_DIR
        self.log_freq = LOG_FREQ
        self.eval_freq = EVAL_FREQ
        self.rmsprop_epsilon = RMSPROP_EPSILON

        self.a3c_gamma    = A3C_GAMMA
        self.actor_loss_weight   = ACTOR_COEF
        self.critic_loss_weight  = CRITIC_COEF
        self.entropy_coeff = ENTROPY_COEF

        self.policy_local  = ActorCritic()

        lr_schedule = PolynomialDecay(self.lr, self.num_episodes*200, end_learning_rate=0)
        self.optimizer = RMSprop(learning_rate=lr_schedule if LR_DECAY else self.lr, epsilon=self.rmsprop_epsilon)

        ## LOGGING MISSING
        
    def reset_memory(self):
        self.obss_batch = np.zeros((self.update_global_freq, *self.obs_dim))
        self.acts_batch = np.zeros((self.update_global_freq, self.num_actions))
        self.rews_batch = np.zeros(self.update_global_freq)

    def replay_memory(self, step, obs, action, reward):
        self.obss_batch[step] = obs
        self.acts_batch[step] = action
        self.rews_batch[step] = reward
        return self.obss_batch, self.acts_batch, self.rews_batch

    def td_target(self, rewards, next_v, done):
        td_targets = np.zeros_like(rewards)
        if done: ret = 0
        else: ret = next_v

        for i in reversed(range(0, len(rewards))):
            ret = rewards[i] + self.A3C_GAMMA * ret
            td_targets[i] = ret
        return td_targets

    def run(self):
        self.total_losses = []
        self.a_losses = []
        self.c_losses = []
        self.e_losses = []

        while A3C_Worker.global_episode < self.num_episodes:  # ???
            self.ep_reward = 0
            self.ep_lengths = 0
            update_counter, done = 0, False
            obs = self.env.reset()
            self.reset_memory()

            while not done:
                action = self.policy_local.act(obs)
                next_obs, reward, done, _ = self.env.step(action.reshape(self.num_actions, ))

                obss, acts, rews = self.replay_memory(update_counter, obs, action, reward)

                self.ep_reward += reward

                if self.update_global_freq <= len(rews) or done:
                    next_obss = np.expand_dims(next_obs, axis=0)

                    next_v = self.policy_local.critic_network(self.policy_local.feature_extractor(next_obss), training=False).numpy()

                    td_targets = self.td_target(rews, next_v, done)
                    td_targets = np.array(td_targets)

                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    acts = tf.constant(acts, tf.float32)
                    advs = tf.constant(advs, tf.float32)
                    td_targets = tf.constant(td_targets, tf.float32)

                    with tf.GradientTape() as tape:
                        mu, var, v_pred = self.policy_local(obss, training=True)

                        a_loss = self.actor_loss(mu, var, acts, advs)
                        c_loss = self.critic_loss(v_pred, td_targets)

                        total_loss = self.actor_loss_weight * a_loss + self.critic_loss_weight * c_loss

                        grad = tape.Gradient(total_loss, self.policy_local.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        grad, _ = tf.clip_by_global_norm(grad, 0.5)
                        self.optimizer.apply_gradients(zip(grad, self.policy_global.trainable_variables))

                        self.policy_local.set_weights(self.policy_global.get_weights())

                    self.total_losses.append(total_loss.numpy())
                    self.a_losses.append(a_loss.numpy())
                    self.c_losses.append(c_loss.numpy())

                    update_counter = 1
                    self.reset_memory()

                self.ep_lengths += 1
                update_counter += 1
                obs = next_obs
            
            A3C_Worker.global_episode += 1
            A3C_Worker.global_total_reward.append(self.ep_reward)
            A3C_Worker.global_total_loss.append(np.mean(self.total_losses))
            A3C_Worker.global_actor_loss.append(np.mean(self.a_losses))
            A3C_Worker.global_critic_loss.append(np.mean(self.c_losses))

            # Calculate Value Across all Workers for CSV
            avg_global_reward = round(np.mean(A3C_Worker.global_total_reward[-self.log_freq:]),3)
            max_global_reward = round(np.max(A3C_Worker.global_total_reward[-self.log_freq:]),3)
            min_global_reward = round(np.min(A3C_Worker.global_total_reward[-self.log_freq:]),3)
            avg_actor_loss    = round(np.mean(A3C_Worker.global_actor_loss[-self.log_freq:]),3)
            avg_critic_loss   = round(np.mean(A3C_Worker.global_critic_loss[-self.log_freq:]),3)

    def actor_loss(self, mu, var, action, adv):
        probability_density_func = tfp.distributions.Normal(mu, var)
        entropy = probability_density_func.entropy()
        log_prob = probability_density_func.log_prob(action)
        expected_value = tf.multiply(log_prob, adv)
        ev_with_entropy = expected_value + entropy * self.entropy_coeff
        
        actor_loss = tf.reduce_sum(-ev_with_entropy)
        return actor_loss

    def critic_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        critic_loss = mse(td_targets, v_pred)
        
        return critic_loss