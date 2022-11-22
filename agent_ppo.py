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

    def reset_memory(self):
        self.replay_memory = {
            "observation" : np.zeros((self.memory_size, ))
        }