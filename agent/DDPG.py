# IMPORTS
import logging
import datetime
import csv
import sys
import gym
import os
import numpy as np
import random
from collections import deque

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Permute, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_model(obs_shape):
    """Identity architecture"""
    model = Sequential()
    model.add(Permute((3,2,1), input_shape=obs_shape))
    model.add(Flatten())
    
    return model

class DDPGAgent():
    """Deep Deterministic Policy Gradient Agent"""

    def __init__(self, params):

        # Configs & Hyperparameters
        self.name = "{}Actions".format(params['agent'])
        self.num_actions = params['num_actions']        
        self.obs_dim = tuple(params['obs_dim'])
        self.epochs = params['num_epochs']
        self.batch_size = params['batch_size']      
        self.memory_size = 12 * self.batch_size
        self.target_steps = 200 * params['num_episodes']
        self.feature_extractor = get_model(self.obs_dim)

        # Replay Buffer
        self.memory = MemoryBuffer(params, self.memory_size)

        # DDPG Hyperparameters; Actor LR should be lower than Critic LR to enable learning
        self.gamma = params['ddpg_gamma']
        self.tau = params['ddpg_tau']
        self.actor_lr = params['ddpg_actor_lr']
        self.critic_lr = params['ddpg_critic_lr']
        self.noise = 0.1                        # Gaussian Noise for explore-exploit
        # self.noise = OUNoise(env.action_space)  # OU Noise
        self.critic_loss_list = []
        self.actor_loss_list = []

        # Networks; Actor, Critic, & Target networks
        self.actor = Actor(params, name='actor')
        self.critic = Critic(params, name='critic')
        self.actor_target = Actor(params, name='actor_target')
        self.critic_target = Critic(params, name='critic_target')

        # Actor and Critic Learning Rate Decay
        actor_lr_schedule = PolynomialDecay(self.actor_lr, self.target_steps, end_learning_rate=0)
        critic_lr_schedule = PolynomialDecay(self.critic_lr, self.target_steps, end_learning_rate=0)
        # Actor and Critic Network Optimizer
        self.actor_optimizer = Adam(learning_rate=actor_lr_schedule if params['lr_decay'] else self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr_schedule if params['lr_decay'] else self.critic_lr)

        # Compile and Optimize
        self.actor.compile(optimizer=self.actor_optimizer)
        self.critic.compile(optimizer=self.critic_optimizer)
        self.actor_target.compile(optimizer=self.actor_optimizer)
        self.critic_target.compile(optimizer=self.critic_optimizer)

        # Soft update of network parameters
        self.update_network_params(tau=self.tau)

        # Manage Logging Properties
        time = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        self.logdir = f"{params['exp_dir']}/log_{time}"
        os.mkdir(self.logdir)

        # Python Logging Gives Easier-to-read Outputs
        logging.basicConfig(filename=self.logdir+'/log.log',
                            format='%(message)s', filemode='w', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        with open(self.logdir + '/log.csv', 'w+', newline='') as file:
            write = csv.writer(file)
            write.writerow(['Episode', 'Ep Reward', 'Avg Reward', 'Best Avg',
                            'Highest Reward', 'Lowest Reward', 'Avg Critic Loss',
                            'Avg Actor Loss'])

        # Load Last Model if Resume is Specified
        if params['resume']:
            weights2load = keras.models.load_model(
                f"{params['exp_dir']}/last_best.model").get_weights()
            self.policy.set_weights(weights2load)
            logging.info("Loaded Weights from Last Best Model!")

    
    def write_log(self, ep, **logs):
        """Write Episode Information to CSV File"""
        line = [ep] + [value for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def update_network_params(self, tau=None):
        """Update Target Actor and Target Critic Network Parameters"""
        if tau is None:
            tau = self.tau
        # Update target actor weights
        weights = []
        target_weights = self.actor_target.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + target_weights[i]*(1-tau))
        self.actor_target.set_weights(weights)

        # Update target critic weights
        weights = []
        target_weights = self.critic_target.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + target_weights[i]*(1-tau))
        self.critic_target.set_weights(weights)


    def remember(self, state, action, reward, next_state, done):
        """Store Transitions"""
        self.memory.add(state, action, reward, next_state, done)


    def save_models(self):
        """Save Model If Average Reward Obtained Is Higher Than Previous Averages"""
        print("Saving / Updating Model...")
        self.actor.save_weights(self.actor.checkpoint_file_train)
        self.actor_target.save_weights(self.actor_target.checkpoint_file_train)
        self.critic.save_weights(self.critic.checkpoint_file_train)
        self.critic_target.save_weights(
            self.critic_target.checkpoint_file_train)


    def save_best(self):
        """Save Model Of Best Score"""
        print("New Best Score! Saving Model...")
        self.actor.save_weights(self.actor.checkpoint_file_best)
        self.actor_target.save_weights(self.actor_target.checkpoint_file_best)
        self.critic.save_weights(self.critic.checkpoint_file_best)
        self.critic_target.save_weights(
            self.critic_target.checkpoint_file_best)


    def load_models(self):
        """Load Model To Use During Test"""
        print("Loading Model...")
        self.actor.load_weights(self.actor.checkpoint_file_train)
        self.actor_target.load_weights(self.actor_target.checkpoint_file_train)
        self.critic.load_weights(self.critic.checkpoint_file_train)
        self.critic_target.load_weights(
            self.critic_target.checkpoint_file_train)
        print("Model Loaded!")


    def load_best(self):
        """Load Best Scoring Model"""
        print("Loading Best Model...")
        self.actor.load_weights(self.actor.checkpoint_file_best)
        self.actor_target.load_weights(self.actor_target.checkpoint_file_best)
        self.critic.load_weights(self.critic.checkpoint_file_best)
        self.critic_target.load_weights(
            self.critic_target.checkpoint_file_best)
        print("Best Model Loaded!")


    def initialize_networks(self, obs):
        """Initialize Networks For Weights To Be Loaded"""
        n_steps = 0
        while n_steps <= self.batch_size:
            action = [1]
            obs_, reward, done = obs, 1, False
            self.remember(np.expand_dims(obs/255, axis=0), action,
                          reward, np.expand_dims(obs_/255, axis=0), done)
            n_steps += 1
        # Initialize Models
        self.train()
        print("Networks Initialized!")


    def select_action(self, obs, env, test_model):
        """Get Action from Actor Network Based on Input Observation"""
        feats = self.feature_extractor(obs)
        action = self.actor(feats)
        # Add Gaussian Noise to Action If Training Model To Improve Explore-Exploit Behavior
        if not test_model:
            action += tf.random.normal(shape=[self.num_actions],
                                       mean=0.0, stddev=self.noise)
        action = tf.clip_by_value(
            action, env.action_space.low, env.action_space.high)

        return action[0]


    def train(self):
        """Train Network"""
        if self.memory.size() < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_batch(
            self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

        # Critic Loss
        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_states)
            critic_value_ = tf.squeeze(self.critic_target(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)    # MSE as Critic Criterion
            self.critic_loss_list.append(critic_loss.numpy())
        
        # Critic Compute Gradients and Apply To Model
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        # Actor Loss
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            self.actor_loss_list.append(actor_loss.numpy())

        # Actor Compute Gradients and Apply To Model
        actor_network_gradient = tape.gradient(
            actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_network_gradient, self.actor.trainable_variables))

        # Soft Update Network Parameters
        self.update_network_params()


    def learn(self, env, params):
        """Run Training Sequence"""
        best_avg = 0
        score_record = []
        # Training Model
        test_model = False
        for ep in range(params['num_episodes']):
            obs = env.reset()
            done = False
            score = 0
            self.step = 0
            self.critic_loss_list = []
            self.actor_loss_list = []

            while not done:
                action = self.select_action(
                    np.expand_dims(obs/255, axis=0), env, test_model)
                obs_, reward, done, info = env.step(action)
                score += reward

                self.remember(np.expand_dims(obs/255, axis=0), action,
                              reward, np.expand_dims(obs_/255, axis=0), done)
                self.train()
                obs = obs_
                self.step += 1

            # Save Best Score
            if ep > 1 and score > max(score_record):
                self.save_best()

            score_record.append(score)
            avg_score = np.mean(score_record[-100:])

            # Save Model Every 20 Episodes
            if ep+1 % 20 == 0:
                self.save_models()

            # Save Model If Current Average Is Greater Than Previous High Average
            if avg_score > best_avg:
                best_avg = avg_score
                self.save_models()

            # Log Information for Each Episode
            self.write_log(ep+1, ep_reward=score, avg_reward=avg_score,
                           avg_best=best_avg, highest_reward=max(score_record),
                           lowest_reward=min(score_record),
                           avg_critic_loss=sum(
                               self.critic_loss_list)/len(self.critic_loss_list),
                           avg_actor_loss=sum(self.actor_loss_list)/len(self.actor_loss_list))

            # Display Episode Information in Console
            print('Ep:', ep+1, "Episode score: %.1f" %
                  score, "Average score: %.1f" % avg_score)


class MemoryBuffer():
    """Store Experiences For Agent To Sample and Learn From"""

    def __init__(self, params, buffer_size):

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.feature_extractor = get_model(params['obs_dim'])


    def add(self, s, a, r, t, s2):
        """Push New Experiences Into Buffer And Increment Counter"""
        s = self.feature_extractor(s)[0]
        t = self.feature_extractor(t)[0]
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)


    def size(self):
        return self.count


    def sample_batch(self, batch_size):
        """Randomly Sample from Memory Buffer According To Batch Size"""
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch


    def clear(self):
        self.buffer.clear()
        self.count = 0


class Critic(Model):
    """Critic Network. Takes Action And Observation as Inputs and Returns Q Value"""

    def __init__(self, params, name='critic'):
        super(Critic, self).__init__()
        # Layer Dimensions
        self.fc1_dims = params['fc_width']
        self.fc2_dims = params['fc_width']

        self.model_name = name
        # Location to Load/Save model
        self.checkpoint_dir = "models/DDPG_train" if params['train'] else params['load_model']
        # Training Model Save
        self.checkpoint_file_train = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg_train.h5')
        # Best Model Save
        self.checkpoint_file_best = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg_best.h5')

        # Build Critic Layers
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)


    def call(self, state, action):
        """Run Forward Pass on Critic Network"""
        concat = tf.concat([state, action], axis=1)
        x = self.fc1(concat)
        x = self.fc2(x)

        # Get Q-Value from evaluating action and observation
        x = self.v(x)

        return x


class Actor(Model):
    """Actor Network. Takes Observation as Inputs and Returns Action"""

    def __init__(self, params, name='actor'):
        super(Actor, self).__init__()
        # Layer Dimensions
        self.fc1_dims = params['fc_width']
        self.fc2_dims = params['fc_width']
        self.n_actions = params['num_actions']

        self.model_name = name
        # Location to Load/Save model
        self.checkpoint_dir = "models/DDPG_train" if params['train'] else params['load_model']
        # Train Model Save
        self.checkpoint_file_train = os.path.join(
            self.checkpoint_dir, self.model_name+'_ddpg_train.h5')
        # Best Model Save
        self.checkpoint_file_best = os.path.join(
            self.checkpoint_dir, self.model_name+'_ddpg_best.h5')

        # Build Actor Layers
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')


    def call(self, state):
        """Run Forward Pass on Actor Network"""
        x = self.fc1(state)
        x = self.fc2(x)

        # Get action from evaluating observation
        x = self.mu(x)

        return x


class OUNoise(object):
    """Ornstein-Ulhenbeck Process Noise to Improve Explore-Exploit Process"""

    def __init__(self, action_space, mean=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        # Hyperparameters
        self.mean = mean
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()


    def reset(self):
        self.state = np.ones(self.action_dim) * self.mean


    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mean - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)

        return np.clip(action + ou_state, self.low, self.high)