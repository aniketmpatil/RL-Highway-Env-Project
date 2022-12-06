import numpy as np
import logging, datetime, os
import sys, csv
from multiprocessing import cpu_count
import threading
from threading import Thread

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Permute
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from racetrack_env import RaceTrackEnv

class ActorCritic(Model):
    '''
        Creates Actor-Critic Networks
    '''
    def __init__(self, params):
        super().__init__('ActorCriticModel')
        self.num_actions = params['num_actions']
        self.fc_layers = params['fc_layers']
        self.fc_width = params['fc_width']
        self.obs_dim = tuple(params['obs_dim'])
        self.create_model()

    def create_model(self):
        '''
            Create Linear Model with a Feature Extractor which is used by 3 networks: Actor Network, 
            Critic Network and Variance Network (used for sampling random actions)
        '''
        self.feature_extractor = Sequential()
        self.feature_extractor.add(Permute((3,2,1), input_shape=self.obs_dim))
        self.feature_extractor.add(Flatten())

        # Take output feature dimensions
        feature_output_dim = self.feature_extractor.layers[-1].output_shape

        # Create three Sequential Networks
        self.actor_network = Sequential()
        self.var_network = Sequential()
        self.critic_network = Sequential()

        # Add Fully Connected Networks
        for _ in range(self.fc_layers):
            self.actor_network.add(Dense(self.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.var_network.add(Dense(self.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
            self.critic_network.add(Dense(self.fc_width, activation='relu', kernel_initializer = 'glorot_uniform'))
        
        # Add Output Layers to all 3 Networks
        self.actor_network.add(Dense(self.num_actions, activation='tanh', kernel_initializer = 'glorot_uniform'))
        self.var_network.add(Dense(self.num_actions, activation='softmax', kernel_initializer = 'glorot_uniform'))
        self.critic_network.add(Dense(1, kernel_initializer = 'glorot_uniform'))

        # Build the model
        self.actor_network.build(feature_output_dim)
        self.var_network.build(feature_output_dim)
        self.critic_network.build(feature_output_dim)

    def call(self, inputs):
        # End-to-end Forward Pass through the networks
        feats = self.feature_extractor(inputs)
        action_output = self.actor_network(feats)
        var_output = self.var_network(feats)
        value_output = self.critic_network(feats)
        return action_output, var_output, tf.squeeze(value_output)
    
    def act(self, obs):
        '''
            Sample Action from the Actor Network
        '''
        obs = np.expand_dims(obs, axis=0)

        feats = self.feature_extractor(obs)
        mu = self.actor_network(feats)
        var = self.var_network(feats)

        # Action is sampled from a Normal Probability Distribution
        probability_density = tfp.distributions.Normal(mu, var)
        action = probability_density.sample(self.num_actions)

        # Action clipped between -1 and 1
        action = tf.clip_by_value(action, -1, 1)

        return action.numpy()


class A3CAgent():
    def __init__(self, params):
        self.policy_global = ActorCritic(params)   # Initialize Global Network
        self.num_workers = cpu_count() if params['num_workers'] == -1 else params['num_workers']

    def learn(self, env, params):
        workers = []
        print("Number of workers: ", self.num_workers)
        for i in range(self.num_workers):
            env = RaceTrackEnv(params)
            workers.append(A3C_Worker(env, self.policy_global, i, params))
    
        for worker in workers: worker.start()    
        for worker in workers: worker.join()

class A3C_Worker(Thread):
    '''
        A3C Worker Class for creating multiple workers to assist the training process
    '''
    global_best = 0
    global_episode = 0
    global_total_reward = []
    global_total_loss = []
    global_actor_loss = []
    global_critic_loss = []
    global_entropy_loss = []
    save_lock = threading.Lock()

    def __init__(self, env, policy_global, worker_idx, params):
        '''
            Initialize the A3C Worker Class for creating multiple workers to assist the training process
        '''
        Thread.__init__(self)
        self.env = env
        self.policy_global = policy_global
        self.num_episodes = params['num_episodes']
        self.num_actions = params['num_actions']
        self.obs_dim = tuple(params['obs_dim'])
        self.update_global_freq = params['update_global_freq']
        self.lr = params['lr']

        self.worker_idx = worker_idx
        self.save_model = params['save_model']
        self.min_reward = params['min_reward']
        self.exp_dir = params['exp_dir']
        self.log_freq = params['log_freq']
        self.eval_freq = params['eval_freq']
        self.rmsprop_epsilon = params['rmsprop_epsilon']

        self.a3c_gamma    = params['a3c_gamma']
        self.actor_coeff   = params['actor_coeff']
        self.critic_coeff  = params['critic_coeff']
        self.entropy_coeff = params['entropy_coeff']

        # Initialize Local Policy Network
        self.policy_local  = ActorCritic(params)

        # Initialize LR Schedule and Optimizer
        lr_schedule = PolynomialDecay(self.lr, self.num_episodes*200, end_learning_rate=0)
        self.optimizer = RMSprop(learning_rate=lr_schedule if params['lr_decay'] else self.lr, epsilon=self.rmsprop_epsilon)

        time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        self.logdir = f"{params['exp_dir']}/log_{time}"
        try:
            os.makedirs(self.logdir)
        except OSError:
            pass

        logging.basicConfig(filename=self.logdir+'/log.log', format='%(message)s', filemode='w', level = logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        # Logging values in CSV file
        with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
            write = csv.writer(file)
            write.writerow(['Total Steps', 'Avg Reward', 'Max Reward', 'Min Reward', 'Avg Actor Loss', 'Avg Critic Loss'])
            
        # Logging Hyperparameters used for training process
        with open(self.logdir + '/opt.txt', 'w+', newline ='') as file:
            # args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
            args = params
            for k, v in sorted(args.items()):
                file.write('  %s: %s\n' % (str(k), str(v)))
        
    def write_log(self):
        '''
            Record to Log File
        '''
        
        logging.info(40*"-")
        logging.info(f"Worker: {self.worker_idx}")
        logging.info(f"Global Steps: {A3C_Worker.global_episode}")
        logging.info(f"Rewards: {self.ep_reward}")
        logging.info(f"Episode Steps: {self.ep_lengths}")

        if len(self.a_losses) > 0 or len(self.c_losses) > 0:
            logging.info(f"Total Loss: {np.mean(self.total_losses):.5f}")
            logging.info(f"Actor Loss: {np.mean(self.a_losses):.5f}")
            logging.info(f"Critic Loss: {np.mean(self.c_losses):.5f}")
            
        logging.info(40*"-")


    def write_csv(self, step, g_mean_rews, g_max_rews, g_min_rews, g_a_loss, g_c_loss):
        '''
            Record to CSV File
        '''

        line = [step, g_mean_rews, g_max_rews, g_min_rews, g_a_loss, g_c_loss]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)


    def reset_memory(self):
        '''
            Reset worker memory
        '''
        self.obss_batch = np.zeros((self.update_global_freq, *self.obs_dim))
        self.acts_batch = np.zeros((self.update_global_freq, self.num_actions))
        self.rews_batch = np.zeros(self.update_global_freq)

    def replay_memory(self, step, obs, action, reward):
        '''
            Add to worker memory for updating Global Network
        '''
        self.obss_batch[step] = obs
        self.acts_batch[step] = action
        self.rews_batch[step] = reward
        return self.obss_batch, self.acts_batch, self.rews_batch

    def td_target(self, rewards, next_v, done):
        '''
            Calculate TD Target values
        '''
        td_targets = np.zeros_like(rewards)
        if done: ret = 0
        else: ret = next_v

        for i in reversed(range(0, len(rewards))):
            ret = rewards[i] + self.a3c_gamma * ret
            td_targets[i] = ret
        return td_targets

    def run(self):
        '''
            This function runs when the actor.start() function is called on a multithreaded implementation
        '''
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

                # Take action and save interaction with the environment with 
                obss, acts, rews = self.replay_memory(update_counter, obs, action, reward)

                self.ep_reward += reward

                if self.update_global_freq <= len(rews) or done:
                    next_obss = np.expand_dims(next_obs, axis=0)

                    next_v = self.policy_local.critic_network(self.policy_local.feature_extractor(next_obss), training=False).numpy()
                    
                    # Calculate TD Target values
                    td_targets = self.td_target(rews, next_v, done)
                    td_targets = np.array(td_targets)

                    values     = self.policy_local.critic_network(self.policy_local.feature_extractor(obss), training=False).numpy()

                    # Calculate Advantage Values
                    advs       = td_targets - values
                    advs       = np.array(advs)
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    acts = tf.constant(acts, tf.float32)
                    advs = tf.constant(advs, tf.float32)
                    td_targets = tf.constant(td_targets, tf.float32)

                    with tf.GradientTape() as tape:
                        mu, var, v_pred = self.policy_local(obss, training=True)

                        # Calculate Actor and Critic Losses
                        a_loss = self.actor_loss(mu, var, acts, advs)
                        c_loss = self.critic_loss(v_pred, td_targets)

                        # Calculate Total loss based on provided coefficients
                        total_loss = self.actor_coeff * a_loss + self.critic_coeff * c_loss

                        # Compute Gradients and applu using Optimizer
                        grad = tape.gradient(total_loss, self.policy_local.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        grad, _ = tf.clip_by_global_norm(grad, 0.5)
                        self.optimizer.apply_gradients(zip(grad, self.policy_global.trainable_variables))

                        # Transfer weights from local policy network to global one
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

            # Write to Log and CSV
            if A3C_Worker.global_episode % self.log_freq == 0:
                self.write_log()
                self.write_csv(A3C_Worker.global_episode, avg_global_reward, max_global_reward, min_global_reward, \
                               avg_actor_loss, avg_critic_loss)

            # Save Model if Episode Reward is better than before
            if self.ep_reward >= np.max([self.min_reward, A3C_Worker.global_best]) and self.save_model:
                with A3C_Worker.save_lock:
                    logging.info(f"Saving Model! Worker: {self.worker_idx}, Episode Score: {self.ep_reward}")
                    self.policy_local.save(f'{self.exp_dir}/R{self.ep_reward:.0f}.model')
                    A3C_Worker.global_best = self.ep_reward

            # Save Model for Every 500 Episodes to Check Training Progress
            if A3C_Worker.global_episode % 500 == 0:
                global_episode = A3C_Worker.global_episode
                with A3C_Worker.save_lock:
                    logging.info(f"Saving Every 500th Model! Worker: {self.worker_idx}, Episode Score: {self.ep_reward}")
                    self.policy_local.save(f'{self.exp_dir}/checkpoint_{global_episode}.model')

    def actor_loss(self, mu, var, action, adv):
        '''
            Function to Compute Actor Network Loss
        '''
        probability_density_func = tfp.distributions.Normal(mu, var)
        entropy = probability_density_func.entropy()
        log_prob = probability_density_func.log_prob(action)
        expected_value = tf.multiply(log_prob, adv)
        ev_with_entropy = expected_value + entropy * self.entropy_coeff
        
        actor_loss = tf.reduce_sum(-ev_with_entropy)
        return actor_loss

    def critic_loss(self, v_pred, td_targets):
        '''
            Function to Compute Critic Network Loss
        '''
        mse = tf.keras.losses.MeanSquaredError()
        critic_loss = mse(td_targets, v_pred)
        
        return critic_loss