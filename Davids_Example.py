#Mario doesn't know what a Goomba is yet
import gym
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import time
def rule_based_agent(observation, info):
    #Extract relevant information from the observation
    player_x = info['x_pos']
    player_y = info['y_pos']
    
    #Define some basic rules
    if player_x > 10:   #Jump if Mario is above the ground
        return 2
    else:
        return 1
def main():
    
    env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    observation = env.reset()
    
    total_reward = 0
    done = False
    
    action = env.action_space.sample()
    while not done:
        time.sleep(0.0000000000001)
        obs, reward, terminated, truncated, info = env.step(action)
        action = rule_based_agent(obs, info)
        total_reward += reward
    
        print("Total reward:", total_reward)
    
if __name__ == "__main__":
    main()
