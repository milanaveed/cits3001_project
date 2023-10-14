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
env = gym_super_mario_bros.make("SuperMarioBros-1-2-v3")

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



#####################################################################
# Load the training result
#####################################################################
model = PPO.load('./reinforcement-learning/train1-2-v3/1-2_v3model_50000')

def test():
    done = True
    number_of_attemps = 0
    while number_of_attemps < 20:
        if done:
            state = env.reset()
            number_of_attemps +=1   
        action, _ = model.predict(state)
        print(ONLY_THREE_MOVEMENT[action[0]])
        state, reward, done, info = env.step(action)
        env.render()
        if info[0]['flag_get']:
            print("########### WINNNNNNNNNNNNN! ###########")
            print(info)
            env.close()
            return
    env.close()
    return

test()

# shortcut for taking screenshots: windows + shift + S


#####################################################################
# Command for Tensorboard
#####################################################################
# cd into the log directory, `tensorboard --logdir=.`


#####################################################################
# Testing the game and show frames
#####################################################################
# state = env.reset()
# state, reward, done, info = env.step([env.action_space.sample()])
'''
VS Code and the Python extension now has TensorBoard integrated in it in its latest release!

To start a TensorBoard session from VSC:

Open the command palette (Ctrl/Cmd + Shift + P)
Search for the command “Python: Launch TensorBoard” and press enter.
You will be able to select the folder where your TensorBoard log files are located. By default, the current working directory will be used.
VSCode will then open a new tab with TensorBoard and its lifecycle will be managed by VS Code as well. This means that to kill the TensorBoard process all you have to do is close the TensorBoard tab.
'''


# def display_4_frames():
#     plt.figure(figsize=(15, 12))
#     for i in range(state.shape[3]):
#         plt.subplot(1, 4, i+1)
#         plt.imshow(state[0][:, :, i])
#     plt.show()

# display_4_frames()

# print(env.step(1)[1])
# print(state.shape)
# plt.imshow(state[0])
# plt.show()

# print(env.observation_space.shape)

# Create a flag - restart or not

# done = True
# # Loop through each frame in the game
# for step in range(1000):
#     # Start the game to begin with
#     if done:
#         # Start the game
#         env.reset()
#     # Do random actions
#     action = env.action_space.sample()
#     print(ONLY_THREE_MOVEMENT[action])
#     state, reward, done, info = env.step(action)
#     # Show the game on the screen
#     env.render()
# # Close the game
# env.close()