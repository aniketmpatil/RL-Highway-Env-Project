import tensorflow as tf 
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import math

LAYER1_SIZE = 512
LAYER2_SIZE = 512
LAYER3_SIZE = 512
LEARNING_RATE = 0.0001
TAU = 0.001

class ActorNetwork:
    def __init__(self, sess, state_dim, action_dim):
        self.time_step = 0
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.state_input,self.action_output,self.net,self.is_training = self.create_network(state_dim,action_dim)

    def create_network(self,state_dim,action_dim):
        layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		layer3_size = LAYER3_SIZE

        state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = self.variable([layer2_size,layer3_size],layer2_size)
		b3 = self.variable([layer3_size],layer2_size)
		W4 = tf.Variable(tf.random_uniform([layer3_size,1],-3e-3,3e-3))
		b4 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

        layer1 = tf.matmul(state_input,W1) + b1
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,W2) + b2
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='batch_norm_2',activation=tf.nn.relu)
		layer3 = tf.matmul(layer2_bn, W3) + b3
		layer3_bn = self.batch_norm_layer(layer3, training_phase=is_training, scope_bn='batch_norm_3',activation=tf.nn.relu)

        action = tf.tanh(tf.matmul(layer3_bn,W4) + b4)

		return state_input,action,[W1,b1,W2,b2,W3,b3,W4,b4],is_training

        # create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.net)