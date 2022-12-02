import tensorflow as tf

from tensorflow import square as sq
from tensorflow import multiply as mul
from tensorflow import reduce_mean as rmean

from tensorflow.train import AdamOptimizer as Adam

from FCNN import FCNN


class Critic:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.input_t = input_t
        self.__dict__.update(params)
        self.__dict__.update(input_t)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self):
        q_input = tf.concat([self.x, self.u], axis=1)
        # MAIN CRITIC NETWORK
        self.q = FCNN(q_input, 1, self.n_layers, self.n_units,
                      tf.nn.relu, name="q")
        # TARGET CRITIC NETWORK
        self.Q = FCNN(q_input, 1, self.n_layers, self.n_units,
                      tf.nn.relu, name="t_q")

    def define_operations(self):
        with tf.name_scope("critic_ops"):
            # LOSS
            loss = tf.sqrt(rmean(sq(self.p - self.q.nn)))
            # MINIMIZE LOSS OP
            self.minimize = Adam(self.lr, name="q_adam")\
                .minimize(loss, var_list=self.q.net_params)
            # ACTION GRADIENTS
            self.action_grads = tf.gradients(self.q.nn, self.u, name="dq_du")
            # UPDATE TARGET OP
            net_param_pairs = zip(self.q.net_params, self.Q.net_params)
            with tf.name_scope("update_target_q"):
                self.updt_Q = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                               for i, j in net_param_pairs]

    def predict(self, x, u):
        return self.session.run(self.q.nn,
                                feed_dict={self.x: x, self.u: u})

    def predict_target(self, x, u):
        return self.session.run(self.Q.nn,
                                feed_dict={self.x: x, self.u: u})

    def train(self, x, u, t):
        return self.session.run([self.q.nn, self.minimize],
                                feed_dict={self.x: x, self.u: u,
                                           self.p: t})

    def get_action_grads(self, x, u):
        return self.session.run(self.action_grads,
                                feed_dict={self.x: x, self.u: u})[0]

    def update_target(self):
        self.session.run(self.updt_Q)