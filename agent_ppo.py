import os
import sys
import csv
import numpy as np
import datetime
import logging

import tensorflow as tf
from tf import keras
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.optimizers.schedules import PolynomialDecay
from keras.losses import MeanAbsoluteError, MeanSquaredError

class AgentPPO():
    def __init__(self):
        self.batch_size = 256
        self.lr = 5e-5
        self.epochs = 10
        self.no_of_actions = 2
        self.epsilon = 0.6
        self.observation_dim = 2
        self.memory_size = 0
        self.target_steps = 200 * 0

        self.GAE_GAMMA = 0
        self.GAE_LAMBDA = 0
        self.PPO_Epsilon = 0
        self.Target_kl = 0
        self.entropy = 0.001

        self.totals_steps = self.num_updates = 0
        self.best = self.eval_best = 0
        self.losses = []

        self.reset_memory()


    def reset_memory(self):
        self.replay_memory = {
            "observation" : np.zeros((self.memory_size, self.observation_dim)),
            "actions" : np.zeros((self.memory_size, self.no_of_actions)),
            "rewards" : np.zeros(self.memory_size),
            "values" : np.zeros(self.memory_size),
            "probabilities" : np.zeros(self.memory_size),
            "done" : np.zeros(self.memory_size),
            "last_ep_starts" : np.zeros(self.memory_size)
        }

    def update_replay(self, step, observation, action, reward, value, probabilities, done, last_ep_start):
        self.replay_memory["observation"][step] = observation
        self.replay_memory["actions"][step] = action
        self.replay_memory["rewards"][step] = reward
        self.replay_memory["values"][step] = value
        self.replay_memory["probabilities"][step] = probabilities
        self.replay_memory["done"][step] = done
        self.replay_memory["last_ep_starts"][step] = last_ep_start

    def collect_rollout(self, env, opt):
        
        episode_rewards = []
        number_of_steps = 0
        self.episode_counter = 0
        self.last_observation = env.reset()

        while number_of_steps != self.memory_size - 1:
            steps = 0
            done = False
            last_episode_start = True
            episode_rewards = 0
            
            while True:

                action, value, logp = self.policy.act(self.last_observation)
                new_observation, reward, done, _  = env.step(action)

                if done or number_of_steps == self.memory_size - 1:
                    self.update_replay(number_of_steps, self.last_observation, action, reward, value, logp, done, last_episode_start)
                    self.episode_counter += 1
                    break

                self.update_replay(number_of_steps, self.last_observation, action, reward, value, logp, done, last_episode_start)
                self.last_observation = new_observation
                last_ep_start = done

                steps += 1
                num_steps += 1
                episode_rewards += reward

                