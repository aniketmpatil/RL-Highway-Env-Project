import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Permute, LSTM, Reshape

from multiprocessing import cpu_count
from threading import Thread

from racetrack_env import RaceTrackEnv

class ActorCritic(Model):
    def __init__(self, opt):
        super().__init__('ActorCriticModel')
        self.create_model(opt)
        self.num_actions = opt.num_actions

    def create_model(self, opt):
        self.feature_extractor = Sequential()
        self.feature_extractor.add(Permute((3,2,1), input_shape=opt.obs_dim))
        self.feature_extractor.add(Flatten())

        feature_output_dim = self.feature_extractor.layers[-1].output_shape

        self.actor_network = Sequential()
        self.var_network = Sequential()
        self.critic_network = Sequential()

        for _ in range(opt.fc_layers):
            self.actor_network.add(Dense(opt.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.var_network.add(Dense(opt.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.critic_network.add(Dense(opt.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
        
        self.actor_network.add(Dense(opt.num_actions, activation='tanh', kernel_initializer = 'glorot_uniform'))
        self.var_network.add(Dense(opt.num_actions, activation='softmax', kernel_initializer = 'glorot_uniform'))
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
    def __init__(self, opt=None):
        self.policy_global = ActorCritic(opt)   # Initialize Global Network
        self.num_workers = cpu_count() if opt.num_workers == None else opt.num_workers

    def learn(self, env, opt):
        workers = []

        for i in range(self.num_workers):
            env = RaceTrackEnv(opt)
            workers.append(A3C_Worker(env, self.policy_global, i, opt))


class A3C_Worker(Thread):
    def __init__(self):
        pass