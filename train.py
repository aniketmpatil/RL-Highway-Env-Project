import gym, os, argparse
import highway_env
import matplotlib.pyplot as plt
# import numpy as np
# from gym.wrappers import Monitor
# import tensorflow.keras as keras

from racetrack_env import RaceTrackEnv
from agent.A3C import A3CAgent

A3C_GAMMA = 0.99
RMSPROP_EPSILON = 1e-5
UPDATE_GLOBAL_FREQ = 5
NUM_AGENTS = None

TRAIN_MODE = True
PRETRAINED_MODEL = None
SAVE_VIDEO = False

args = {"A3C_GAMMA": A3C_GAMMA,
        "RMSPROP_EPSILON": RMSPROP_EPSILON,
        "UPDATE_GLOBAL_FREQ": UPDATE_GLOBAL_FREQ,
        "NUM_AGENTS": NUM_AGENTS,
        "PRETRAINED_MODEL": PRETRAINED_MODEL
        }

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # Configuration Settings
        self.parser.add_argument('--mode', default='train', help='Train, Test, Manual')
        self.parser.add_argument('--agent', default='PPO', help='DQN, DDPG, A3C, PPO')
        self.parser.add_argument('--exp_id', default='default', help='Unique Experiment Name for Saving Logs & Models')
        self.parser.add_argument('--resume', action='store_true', help='Whether to Load Last Model for Further Training')
        self.parser.add_argument('--load_model', default=None, help='Model to load for Testing')
        self.parser.add_argument('--save_model', default=True, help='Whether to Save Model during Training')
        self.parser.add_argument('--save_video', action='store_true', help='Saves Env Render as Video')

        # Neural Network Settings
        self.parser.add_argument('--arch', default='Identity', help='Neural Net Backbone')
        self.parser.add_argument('--fc_layers', default=2, type=int, help='Number of Dense Layers')
        self.parser.add_argument('--fc_width', default=256, type=int, help='Number of Channels in Dense Layers')

        # Problem Space Settings
        self.parser.add_argument('--obs_dim', default=(2,18,18), type=int, nargs=3, help='Agent Observation Space')
        self.parser.add_argument('--num_actions', default=1, type=int, help='Agent Action Space')
        self.parser.add_argument('--offroad_thres', default=-1, type=int, help='Number of Steps Agent is Allowed to Ride Offroad')
        self.parser.add_argument('--spawn_vehicles', default=0, type=int, help='Number of Non-Agent Vehicles to Spawn, Set 0 to Disable')
        self.parser.add_argument('--all_random', action='store_true', help='Whether to Train on All Random Vehicles')
        self.parser.add_argument('--random_lane', action='store_true', help='Whether to Randomize Agent Spawn Lane')
        self.parser.add_argument('--random_obstacles', default=0, type=int, help='Number of Static Obstacles to Spawn (Unused)')

        # Experiment Settings
        self.parser.add_argument('--num_episodes', default=10000, type=int, help='Number of Episodes to Train')
        self.parser.add_argument('--log_freq', default=20, type=int, help='Frequency of Logging (Episodes)')
        self.parser.add_argument('--eval_freq', default=100, type=int, help='Frequency to Run Evaluation Runs')
        self.parser.add_argument('--min_reward', default=50, type=int, help='Minimum Reward to Save Model')

        # Hyperparameters
        self.parser.add_argument('--lr', default=5e-4, type=float, help='Policy Learning Rate')
        self.parser.add_argument('--lr_decay', action='store_true', help='Whether to Decay Learning Rate')
        self.parser.add_argument('--batch_size', default=64, type=int, help='Policy Update Batch Size')
        self.parser.add_argument('--num_epochs', default=10, type=int, help='Num Epochs for Policy Gradient')

        # DQN Hyperparameters
        self.parser.add_argument('--epsilon', default=1, type=float, help='Initial Value of Epsilon')
        self.parser.add_argument('--epsilon_decay', default=0.9995, type=float, help='Decay Ratio of Epsilon')
        self.parser.add_argument('--min_epsilon', default=0, type=float, help='Minimum Value of Epsilon')
        self.parser.add_argument('--dqn_gamma', default=0.99, type=float, help='Frequency of Updating Target Model')
        self.parser.add_argument('--update_freq', default=20, type=int, help='Frequency of Updating Target Model')
        self.parser.add_argument('--replay_size', default=10000, type=int, help='Size of the Replay Memory Buffer')
        self.parser.add_argument('--min_replay_size', default=500, type=int, help='Minimum Memory Entries before Training')

        # PPO Hyperparameters
        self.parser.add_argument('--gae_lambda', default=0.95, type=float, help='Generalised Advantage Estimate Lambda')
        self.parser.add_argument('--gae_gamma', default=0.9, type=float, help='Generalised Advantage Estimate Gamma')
        self.parser.add_argument('--ppo_epsilon', default=0.2, type=float, help='Clipping Loss Epsilon')
        self.parser.add_argument('--target_kl', default=None, type=float, help='Max KL Divergence for Training Sequence')
        self.parser.add_argument('--ppo_memory_size', default=2048, type=int, help='Size of Replay Buffer for PPO')
        
        # A3C Hyperparameters
        self.parser.add_argument('--a3c_gamma', default=0.99, type=float, help='Generalised Advantage Estimate Gamma')
        self.parser.add_argument('--rmsprop_epsilon', default=1e-5, type=float, help='RMSProp epsilon')
        self.parser.add_argument('--update_global_freq', default=5, type=int, help='Frequency of Updating Master Agent')
        self.parser.add_argument('--num_workers', default=None, type=int, help='Number of Workers')
        
        # DDPG Hyperparameters
        # self.parser.add_argument('--ddpg_actor_lr', default=0.001, type=float, help='DDPG Actor Learning Rate')
        # self.parser.add_argument('--ddpg_critic_lr', default=0.002, type=float, help='DDPG Critic Learning Rate')
        # self.parser.add_argument('--ddpg_gamma', default=0.99, type=float, help='Generalised Advantage Estimate Gamma')
        # self.parser.add_argument('--ddpg_tau', default=0.005, type=float, help='Soft Update Coefficient')
        # self.parser.add_argument('--ddpg_best', default=False, type=bool, help='Get Best Scoring Model For DDPG')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        
        if not os.path.exists("./logs"):
            os.mkdir("./logs/")
        
        opt.exp_dir = f"./logs/{opt.exp_id}"
        if not os.path.exists(opt.exp_dir):
            os.mkdir(opt.exp_dir)

        return opt

if __name__ == "__main__":
    opt = opts().parse()
    print(opt)

    env = RaceTrackEnv(opt)
    # env = gym.make('racetrack-v0')
    # env = gym.make("racetrack-v0")
    env.reset()

    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
    
    plt.imshow(env.render(mode="rgb_array"))
    plt.show()

    # if SAVE_VIDEO:
    #     env = Monitor(env, f'./outputs/A3C_{PRETRAINED_MODEL}/', force=True)

    # if TRAIN_MODE:
    #     agent = A3CAgent(args)
    #     agent.learn(env, args)
    # else:
    #     total_reward, obs, done, seq = 0, env.reset(), False, []

    #     model = keras.models.load_model(PRETRAINED_MODEL)

    #     while not done:
    #         action = model(np.array([obs]))[0]
    #         # obs, reward, done, info

