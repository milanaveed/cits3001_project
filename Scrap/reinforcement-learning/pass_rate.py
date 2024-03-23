# Import the game
import gym
import gym_super_mario_bros
# Import keyboard control to ignore keyboard interruption and prevent accidental quits
import os
import signal
# Path management for saving models and log files
from pathlib import Path
# Import numpy for calculation and visualisation
import numpy as np
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import the GrayScaling Wrapper and Resize Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation
# Import the Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import PPO algorithm and callback function
from stable_baselines3 import PPO
import torch as th
from torch import nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# Visualisation tool
from matplotlib import pyplot as plt

# Reference: https://www.youtube.com/watch?v=2eeYqJ0uBKE

###########################################################################

class SkipFrame(gym.Wrapper):
    ''''
    Define SkipFrame (https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo/notebook)
    '''
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    

class CustomRewardAndDoneEnv(gym.Wrapper):
    '''
    Define reward function
    '''
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        # self.current_score = 0
        self.current_x = 0  # Record current x position
        self.current_x_count = 0  # Count the consecutive occurrences of current x position
        self.max_x = 0  # Record the right most x position mario has reached

    def reset(self, **kwargs):  # Reset the environment to its initial state and resets the internal state variables of the custom wrapper
        # self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)  # Obtain reward for the current action

        # If moves right, add reward to encourage rightward movement
        reward += max(0, info['x_pos'] - self.max_x)

        # Checks if mario stays at the same x position
        # if (info['x_pos'] - self.current_x) == 0:
        #     self.current_x_count += 1
        #     if self.current_x_count > 20:  # If stucks at the same x position, apply penalty
        #         reward -= 500
        #         done = True
        # else:
        #     self.current_x_count = 0 

        if info["flag_get"]:  # If touches the flag, apply reward, and alter the done flag to end the episode
            reward += 500
            done = True
            print("REACH GOAL")

        if info["life"] < 2:  # If dies, apply death penalty, and alter the done flag to end the episode
            reward -= 500
            done = True

        # self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)  # Update the right most x position value
        self.current_x = info["x_pos"]  # Update the current x position value
        return state, reward / 10., done, info


############################################################################
# Set up game environment
############################################################################
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")

# * Simplify actions -- Option 1: Use SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# print("Available movements: ", SIMPLE_MOVEMENT)

# * Simplify actions -- Option 2: Sample the ['right'] and ['right', 'A'] actions
# ONLY_RIGHT_MOVEMENT = SIMPLE_MOVEMENT[1:3]
# env = JoypadSpace(env, ONLY_RIGHT_MOVEMENT)
# print("Available movements: ", ONLY_RIGHT_MOVEMENT)   # [['right'], ['right', 'A']]

# * Simplify actions -- Option 3: ONLY_THREE_MOVEMENT = [['left'], ['right', 'B'], ['right', 'A', 'B']]
ONLY_THREE_MOVEMENT = [['right', 'B'], ['right', 'A', 'B'], ['left', 'A'] ]
env = JoypadSpace(env, ONLY_THREE_MOVEMENT)
print("Available movements: ", ONLY_THREE_MOVEMENT)

print(env.observation_space.shape)  # (240, 256, 3)

# Reset reward function
env = CustomRewardAndDoneEnv(env)

# Skip frames
env = SkipFrame(env, skip=4)
print(env.observation_space.shape)

# Grayscale environment
env = GrayScaleObservation(env, keep_dim=True)
print(env.observation_space.shape)  # (240, 256, 1)

# Resize observation
env = ResizeObservation(env, shape=84)
print(env.observation_space.shape)  # (84, 84, 1)

# Create a vectorized wrapper for an environment
env = DummyVecEnv([lambda: env])

# # Stack the four frames each time
env = VecFrameStack(env, 4, channels_order='last')
print(env.observation_space.shape)  # (1, 84, 84, 4)

###############################################################################
# Add keyboard control and prevent accidental quits when training on Windows
###############################################################################
current_pid=os.getpid()
print(current_pid)

def SigIntHand(SIG, FRM):
    print("Ctrl-C does not work on the cmd prompt.")
    print("List the process id by `tasklist | findstr python`")
    print("COMMAND THAT CAN KILL THE PROCESS:")
    print(f"taskkill /PID {current_pid} /F")

signal.signal(signal.SIGINT, SigIntHand)


#####################################################################
# Create a log file to save the performance metrics' data
#####################################################################
DIR = './reinforcement-learning/train1-1-v302/'
os.makedirs(DIR, exist_ok=True)

# DISTANCE_LOG_PATH = DIR + 'x_position_log.csv'
PASS_RATE_LOG_PATH = DIR + 'pass_rate_log.csv'

# with open(DISTANCE_LOG_PATH, 'a') as f:  
#     print('timesteps_of_model,average_distance,best_distance', file=f)

with open(PASS_RATE_LOG_PATH, 'a') as f:  
    print('timesteps_of_model,pass_rate(%)', file=f)


#####################################################################
# Load the training result
#####################################################################
# Set test parameters
NUMBER_OF_TRIALS = 100
MAX_ACTION_NUMBER = 500


def get_pass_rate():
    #! 可以接着训练，不用从头开始
    for time_steps_of_model in range(50000, 10000001, 50000):
        model = PPO.load('./reinforcement-learning/train1-1-v302/1-1_v302model_{}'.format(time_steps_of_model))

        # x_position = [0] * NUMBER_OF_TRIALS
        # best_x_position = 0
        number_of_pass = 0

        for i in range(NUMBER_OF_TRIALS):
            state = env.reset()  
            done = False
            while not done:
                action, _ = model.predict(state)
                # print(ONLY_THREE_MOVEMENT[action[0]])
                state, reward, done, info = env.step(action)
                # x_position[i] = info[0]['x_pos']
                # env.render()

                if info[0]['flag_get']:
                    number_of_pass += 1
                    # print("########### WINNNNNNNNNNNNN! ###########")

            # if x_position[i] > best_x_position:
            #     best_x_position = x_position[i]

        print('####################################################')
        print('time steps of current model:', time_steps_of_model, '/', 10000000)
        # print('average distance:', np.mean(x_position),
        #         'best_distance:', best_x_position)
        print('number_of_pass:', number_of_pass)
        print('####################################################')


        # with open(DISTANCE_LOG_PATH, 'a') as f:
        #         print(time_steps_of_model, ',', np.mean(x_position), ',', best_x_position, file=f)

        with open(PASS_RATE_LOG_PATH, 'a') as f:
                print(time_steps_of_model, ',', number_of_pass, file=f)

    env.close()
    

get_pass_rate()

# shortcut for taking screenshots: windows + shift + S
