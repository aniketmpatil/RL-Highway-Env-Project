import tensorflow as tf
import numpy as np
import numpy.random as nr
import math

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

		# create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

		# define training rules
		self.create_training_method()

		self.sess.run(tf.initialize_all_variables())

		self.update_target()
		#self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

	def create_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
		action_output = tf.tanh(tf.matmul(layer2,W3) + b3)

		return state_input,action_output,[W1,b1,W2,b2,W3,b3]

	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
		action_output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])

		return state_input,action_output,target_update,target_net

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]


	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


import tensorflow as tf 
import numpy as np
import math


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		# create q network
		self.state_input,\
		self.action_input,\
		self.q_value_output,\
		self.net = self.create_q_network(state_dim,action_dim)

		# create target q network (the same structure with q network)
		self.target_state_input,\
		self.target_action_input,\
		self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

		self.create_training_method()

		# initialization 
		self.sess.run(tf.initialize_all_variables())
			
		self.update_target()

	def create_training_method(self):
		# Define training optimizer
		self.y_input = tf.placeholder("float",[None,1])
		weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_q_network(self,state_dim,action_dim):
		# the layer size could be changed
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
		W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
		b2 = self.variable([layer2_size],layer1_size+action_dim)
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
		b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)

		return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

	def create_target_q_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])

		return state_input,action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={
			self.y_input:y_batch,
			self.state_input:state_batch,
			self.action_input:action_batch
			})

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

