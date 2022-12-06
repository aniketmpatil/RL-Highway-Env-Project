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
    params_file_path = "config/params.yaml"
    params = read_config(params_file_path)

    agent_name = params['agent']
    train = params['train']

    env = RaceTrackEnv(params)

    if params['save_video']:
        exp_id = params['exp_id']
        env = Monitor(env, f'./videos/{agent_name}_{exp_id}/', force=True)

    if train:
        print("---------- Training ", agent_name, "----------")
        if agent_name == "A3C":
            agent = A3CAgent(params)
            agent.learn(env, params)
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