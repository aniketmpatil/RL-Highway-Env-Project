from gym.wrappers import Monitor
import yaml, os, argparse
import numpy as np
import tensorflow.keras as keras

## Import Environment
from racetrack_env import RaceTrackEnv

## Import all agents
from agent.A3C import A3CAgent
from agent.DDPG import DDPGAgent
from agent.PPO import PPO

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--exp_id', default='default', help='Unique Experiment Name for Saving Logs & Models')
        
        self.parser.add_argument('--obs_dim', default=(2,18,18), type=int, nargs=3, help='Agent Observation Space')
        self.parser.add_argument('--num_actions', default=1, type=int, help='Agent Action Space')
        self.parser.add_argument('--all_random', action='store_true', help='Whether to Train on All Random Vehicles')
        self.parser.add_argument('--spawn_vehicles', default=0, type=int, help='Number of Non-Agent Vehicles to Spawn, Set 0 to Disable')
        self.parser.add_argument('--random_lane', action='store_true', help='Whether to Randomize Agent Spawn Lane')
        self.parser.add_argument('--offroad_thres', default=-1, type=int, help='Number of Steps Agent is Allowed to Ride Offroad')
        
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt

def read_config(file_path):
    '''
        Function to load Hyperparameters
    '''
    with open(file_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    if not os.path.exists("./logs"):
        os.mkdir("./logs/")
    
    data['exp_dir'] = f"./logs/{data['exp_id']}"
    if not os.path.exists(data['exp_dir']):
        os.mkdir(data['exp_dir'])
    return data

if __name__ == "__main__":
    opt = opts().parse()

    params_file_path = "config/params.yaml"
    params = read_config(params_file_path)

    agent_name = params['agent']
    train = params['train']

    env = RaceTrackEnv(opt)

    if params['save_video']:
        exp_id = params['exp_id']
        env = Monitor(env, f'./videos/{agent_name}_{exp_id}/', force=True)

    if train:
        print("---------- Training ", agent_name, "----------")
        if agent_name == "A3C":
            agent = A3CAgent(params)
            agent.learn(env, opt, params)
        elif agent_name == "PPO":
            agent = PPO(params)
            agent.learn(env, params)
        elif agent_name == "DDPG":
            agent = DDPGAgent(params)
            agent.learn(env, params)
    
    else:
        total_reward, obs, done, seq = 0, env.reset(), False, []

        if agent_name == "DDPG":
            agent = DDPGAgent(params)
                        
            agent.initialize_networks(obs)
            if params['ddpg_best'] == True:
                agent.load_best()
            else:
                agent.load_models()
            
            while not done:
                action = agent.select_action(np.expand_dims(obs/255, axis=0), env, test_model=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                print(reward)
            print("Total Reward:", total_reward)

        else:
            model = keras.models.load_model(params['load_model'])
                     
            while not done:
                action = model(np.array([obs]))[0]
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                print(reward)
            print("Total Reward: ", total_reward)