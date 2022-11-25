import os
import sys
import csv
import numpy as np
import datetime
import logging
# import tensorflow.keras as K
import tensorflow as tf
from tf import keras 
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.optimizers.schedules import PolynomialDecay
from keras.losses import MeanAbsoluteError, MeanSquaredError
import keras as K
from .models import get_model

class AgentPPO():
    def __init__(self, opt = None):

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

        self.policy = PolicyModel(opt)
        self.val_loss = MeanSquaredError()
        lr_schedule = PolynomialDecay(self.lr, self.target_steps // self.memory_size * 12 * self.epochs, end_learning_rate=0)
        self.optimizer = Adam(learning_rate=lr_schedule if opt.lr_decay else self.lr)

        self.totals_steps = self.num_updates = 0
        self.best = self.eval_best = 0
        self.losses = []

        # Initialise Replay Memory Buffer
        self.reset_memory()

        # ----------------------------------------------------------------------------------------------------------------------

        # Manage Logging Properties
        time = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        self.logdir = f"{opt.exp_dir}/log_{time}"
        os.mkdir(self.logdir)
        
        # Python Logging Gives Easier-to-read Outputs
        logging.basicConfig(filename=self.logdir+'/log.log', format='%(message)s', filemode='w', level = logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        with open(self.logdir + '/log.csv', 'w+', newline ='') as file:
            write = csv.writer(file)
            write.writerow(['Total Steps', 'Avg Reward', 'Min Reward', 'Max Reward', 'Eval Reward', 'Avg Ep Length'])
        
        with open(self.logdir + '/opt.txt', 'w+', newline ='') as file:
            args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
            for k, v in sorted(args.items()):
                file.write('  %s: %s\n' % (str(k), str(v)))
                
        # Load Last Model if Resume is Specified
        if opt.resume:
            weights2load = K.models.load_model(f'{opt.exp_dir}/last_best.model').get_weights()
            self.policy.set_weights(weights2load)
            logging.info("Loaded Weights from Last Best Model!")

        # -----------------------------------------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------------
    
    def write_log(self, step, **logs):
        """Write Episode Information to CSV File"""
        line = [step] + [round(value, 3) for value in logs.values()]
        with open(self.logdir + '/log.csv', 'a', newline ='') as file:
            write = csv.writer(file)
            write.writerow(line)

    def callback(self, avg_reward):
        """Write Training Information to Console"""
        
        self.avg_ep_len = self.memory_size / self.episode_counter
        
        logging.info(40*"-")
        logging.info(f"Total Steps: {self.total_steps}")
        logging.info(f"Average Train Reward: {avg_reward:.3f}")
        logging.info(f"Average Train Ep Length: {self.avg_ep_len:.3f}")
        logging.info(f"Average Eval Reward: {self.eval_reward:.3f}")
        logging.info(f"Num. Model Updates: {self.num_updates}")
        
        if len(self.losses) > 0:
            logging.info(f"Total Loss: {np.mean(self.losses):.5f}")
            logging.info(f"Actor Loss: {np.mean(self.a_losses):.5f}")
            logging.info(f"Critic Loss: {np.mean(self.c_losses):.5f}")
            logging.info(f"Entropy Loss: {np.mean(self.e_losses):.5f}")
            logging.info(f"Approx KL Div: {np.mean(self.kl_divs):.3f}")
            logging.info(f"Explained Var: {self.explained_var:.3f}")
            logging.info(f"Policy Std Dev: {np.exp(self.policy.log_std.numpy()).squeeze():.3f}")
            logging.info(f"Learning Rate: {self.optimizer._decayed_lr('float32').numpy():.8f}")
            
        logging.info(40*"-")

    # ----------------------------------------------------------------------------------------------
    

    def learn(self, env, opt):
        # Run rollout and training sequence
        while self.total_steps < self.target_steps:
            self.collect_rollout(env, opt)
            self.train()
        self.policy.save(f'{opt.exp_dir}/model_last.model')

    def reset_memory(self):
        self.replay_memory = {
            "observation" : np.zeros((self.memory_size, self.observation_dim)),
            "actions" : np.zeros((self.memory_size, self.no_of_actions)),
            "rewards" : np.zeros(self.memory_size),
            "values" : np.zeros(self.memory_size),
            "probabilities" : np.zeros(self.memory_size),
            "done" : np.zeros(self.memory_size),
            "last_episode_starts" : np.zeros(self.memory_size)
        }

    def update_replay(self, step, observation, action, reward, value, probabilities, done, last_episode_start):
        self.replay_memory["observation"][step] = observation
        self.replay_memory["actions"][step] = action
        self.replay_memory["rewards"][step] = reward
        self.replay_memory["values"][step] = value
        self.replay_memory["probabilities"][step] = probabilities
        self.replay_memory["done"][step] = done
        self.replay_memory["last_episode_starts"][step] = last_episode_start

    def collect_rollout(self, env, opt):
        # Collect experiences and store them"

        episode_rewards = []
        number_of_steps = 0
        self.episode_counter = 0
        self.last_observation = env.reset()

        while number_of_steps != self.memory_size - 1:

            steps = 0
            done = False
            last_episode_start = True
            episode_reward = 0
            
            while True:

                # Get the Environment
                action, value, logp = self.policy.act(self.last_observation)
                
                # Get the observation and reward from step function
                new_observation, reward, done, _  = env.step(action)

                # Check if buffer size is full, end the step
                if done or number_of_steps == self.memory_size - 1:
                    self.update_replay(number_of_steps, self.last_observation, action, reward, value, logp, done, last_episode_start)
                    self.episode_counter += 1
                    break

                # Update the replay buffer
                self.update_replay(number_of_steps, self.last_observation, action, reward, value, logp, done, last_episode_start)
                self.last_observation = new_observation
                last_episode_start = done

                # increment counters
                steps += 1
                number_of_steps += 1
                episode_reward += reward

            # Reset the last observation
            self.last_observation = env.reset()

            # store the episode reward
            episode_rewards.append(episode_reward)

        # calculate last value for finished episode
        _, self.last_value, _ = self.policy.act(new_observation)
        self.totals_steps += number_of_steps + 1

        average_reward = np.mean(episode_rewards)
        minimum_reward = np.min(episode_rewards)
        maximum_reward = np.max(episode_rewards)

        # Run one evaluation run task
        observation = env.reset()
        reward = 0
        done = False

        while not done:
            action, _ = self.policy(np.expand_dims(observation, axis = 0))
            observation, r, done, _ = env.step(action)
            reward += r
        self.eval_reward = reward

        # -------------------------------------------------------------------------------------------------------

        # Show Training Progress on Console
        self.callback(average_reward)

        self.write_log(self.total_steps, reward_avg=average_reward, reward_min=minimum_reward,
                       reward_max=maximum_reward, eval_reward=self.eval_reward, avg_ep_len=self.avg_ep_len)

        # Save Model if Average Reward is Greater than a Minimum & Better than Before
        if average_reward >= np.max([opt.minimum_reward, self.best]) and opt.save_model:
            self.best = average_reward
            self.policy.save(f'{opt.exp_dir}/R{average_reward:.0f}.model')
        
        # Save Model if Eval Reward is Greater than a Minimum & Better than Before
        if self.eval_reward >= np.max([opt.minimum_reward, self.eval_best]) and opt.save_model:
            self.eval_best = self.eval_reward
            self.policy.save(f'{opt.exp_dir}/eval_R{self.eval_reward:.0f}.model')
        
        # Save Model Every 20 PPO Update Iterations
        if self.total_steps % (20 * self.memory_size) == 0:
            self.policy.save(f'{opt.exp_dir}/checkpoint_{self.total_steps}.model')

        # ------------------------------------------------------------------------------------------------------------


    def process_replay(self, mem):
        # process episode information for values and advantages

        #  get the values, rewards, probabilities, done
        values = mem["values"]
        rewards = mem["rewards"]
        probabilities = mem["probabilities"]
        done = mem["done"][-1]
        last_episode_starts = mem["last_episode_starts"]

        g = self.GAE_GAMMA
        l = self.GAE_LAMBDA

        #  Initialize advantages and return arrays
        advantages = np.empty((self.memory_size,))
        returns = np.empty((self.memory_size,))

        last_advantage = 0

        for step in reversed(range(self.memory_size)):
            if step == self.memory_size - 1:
                next_non_terminal = 1.0 - done
                next_value = self.last_value.squeeze()
            else:
                next_non_terminal = 1.0 - last_episode_starts[step + 1]
                next_value = values[step + 1]
            delta = rewards[step] + g * next_value * next_non_terminal - values[step]
            last_advantage = delta + g * l * next_non_terminal * last_advantage
            advantages[step] = last_advantage

        returns = advantages + values

        return mem["observations"], mem["actions"], advantages, returns, values, probabilities

    def train(self):
        # Training the agent using the collected memory buffer

        # Process returns and advantages for buffer info
        buffer_observations, buffer_actions, buffer_advantages, buffer_returns, buffer_values, buffer_probabilities = self.process_replay(self.replay_memory)
        
        # Generate indices for sampling
        index = np.random.permutation(self.memory_size)

        self.losses = []
        self.a_losses = []
        self.c_losses = []
        self.e_losses = []
        self.kl_divs = []

        for epoch in range(self.epochs):
            for batch_index in range(0, self.memory_size, self.batch_size):
                
                # Go through buffer batch size at a time
                observations = buffer_observations[index[batch_index:batch_index + self.batch_size]]
                actions = buffer_actions[index[batch_index:batch_index + self.batch_size]]
                advantages = buffer_advantages[index[batch_index:batch_index + self.batch_size]]
                returns = buffer_advantages[index[batch_index:batch_index + self.batch_size]]
                probabilities = buffer_probabilities[index[batch_index:batch_index + self.batch_size]]

                # Normalise Advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # cast constant inputs to tensors
                actions = tf.constant(actions, tf.float32)
                advantages = tf.constant(advantages, tf.float32)
                returns = tf.constant(returns, tf.float32)
                probabilities = tf.constant(probabilities, tf.float32)

                with tf.GradientTape() as tape:

                    # Run forward pass on the model and get the new log probabilities
                    a_pred, v_pred = self.policy(observations, training = True)
                    new_log_probs = self.policy.logp(a_pred, actions)

                    # calculate the ratio between the old and new policy
                    ratios = tf.exp(new_log_probs - probabilities)

                    # Clipped Actor Loss
                    loss1 = advantages * ratios
                    loss2 = advantages * tf.clip_by_value(ratios, 1 - self.PPO_EPSILON, 1 + self.PPO_EPSILON)
                    a_loss = tf.reduce_mean(-tf.math.minimum(loss1, loss2))

                    # Entropy Loss
                    entropy = self.policy.entropy()
                    e_loss = -tf.redcue_mean(entropy)

                    # Value Loss
                    c_loss = self.val_loss(returns, v_pred)

                    total_loss = 0.5 * c_loss * a_loss * self.ENTROPY * e_loss

                # Compute KL Divergence for early stopping before backprop
                kl_div = 0.5 * tf.reduce_mean(tf.square(new_log_probs - probabilities))

                if self.Target_kl != None and kl_div > self.Target_kl:
                    logging.info(f"Early stopping at epoch {epoch+1} due to reaching max kl: {kl_div:.3f}")
                    break

                # Compute gradients and apply to model
                gradients = tape.gradient(total_loss, self.policy.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

                # logging
                self.losses.append(total_loss.numpy())
                self.a_losses.append(a_loss.numpy())
                self.c_losses.append(c_loss.numpy())
                self.e_losses.append(e_loss.numpy())
                self.kl_divs.append(kl_div)

                self.num_updates += 1

        self.explained_var = explained_variance(buffer_values,buffer_returns)
        
        self.reset_memory()

class PolicyModel(Model):

    def __init__(self, opt):
        super().__init__('PolicyModel')
        self.build_model(opt)
        self.log_std = tf.Variable(initial_value = -0.5 * np.ones(opt.num_actions, dtype = np.float32))

    def build_model(self, opt):
        
        self.feature_extractor = get_model(opt)

        # Retrieve post-feature extractor dimensions
        for layer in self.feature_extractor.layers:
            feature_output_dim = layer.output_shape

        # define the actor and critic networks
        self.actor_network = Sequential()
        self.critic_network = Sequential()

        for i in range(opt.fc_layers):
            self.actor_network.add(Dense(opt.fc_width, activation='tanh'))
            self.critic_network.add(Dense(opt.fc_width, activation='tanh'))
        
        self.actor_network.add(Dense(opt.num_actions, activation = 'tanh'))
        self.critic_network.add(Dense(1))

        self.actor_network.build(feature_output_dim)
        self.critic_network.build(feature_output_dim)

    def call(self, inputs):
        "Run forward pass for training"
        feats = self.feature_extractor(inputs)
        action_output = self.actor_network(feats)
        value_output = self.critic_network(feats)
        return action_output, tf.squeeze(value_output)
    
    def act(self,observation):
        # Get actions, values and probabilities during experience collection

        observation = np.expand_dims(observation, axis = 0)

        # Run forward passes
        feats = self.feature_extractor(observation)
        a_pred = self.actor_network(feats)
        v_pred = self.critic_network(feats)

        # Calculate log probabilities
        std = tf.exp(self.log_std)
        action = a_pred + tf.random.normal(tf.shape(a_pred)) * std
        action = tf.clip_by_value(action, -1, 1)
        logp_t = self.logp(action, a_pred)
        return action.numpy(), v_pred.numpy().squeeze(), logp_t.numpy().squeeze()

    def logp(self, x, meu):
        # Return log probabilities of action given distribution Parameters
        pre_sum = -0.5 * (((x - meu) / (tf.exp(self.log_std) + 1e-8))**2 + 2 * self.log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis = -1)

    def entropy(self):
        # Return entropy of policy distribution
        entropy = tf.reduce_sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis = -1)
        return entropy

# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()







































    