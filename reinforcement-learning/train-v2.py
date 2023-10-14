# Import the game
import gym
import gym_super_mario_bros
# Import keyboard control to ignore keyboard interruption and prevent accidental quits
import os
import signal
from pathlib import Path
import datetime
from pytz import timezone
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
from stable_baselines3.common.callbacks import BaseCallback
# Visualisation tool
from matplotlib import pyplot as plt

# Reference: https://www.youtube.com/watch?v=2eeYqJ0uBKE

###########################################################################
# Define SkipFrame (https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo/notebook)


class SkipFrame(gym.Wrapper):
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

# Define reward function


class CustomRewardAndDoneEnv(gym.Wrapper):
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
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
            if self.current_x_count > 10:  # If stucks at the same x position, apply penalty
                reward -= 500
        else:
            self.current_x_count = 0 

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
env = gym_super_mario_bros.make("SuperMarioBros-1-2-v0")

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
cur_pid=os.getpid()
print(cur_pid)

def SigIntHand(SIG, FRM):
    print("Ctrl-C does not work on the cmd prompt.")
    print("List the process id by `tasklist | findstr python`")
    print("COMMAND THAT CAN KILL THE PROCESS:")
    print(f"taskkill /PID {cur_pid} /F")

signal.signal(signal.SIGINT, SigIntHand)


##############################################################################
# Train and save the training process
##############################################################################
# Manage files and directories
DATA_DIR = './reinforcement-learning/train1-2-02/'
os.makedirs(DATA_DIR, exist_ok=True)
LOG_DIR = './reinforcement-learning/logs1-2-02/'
REWARD_LOG_PATH = DATA_DIR + 'reward_log.csv'
X_POSITION_LOG_PATH = DATA_DIR + 'x_position_log.csv'

# Set test parameters
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000

# Add a header line into the csv log file
with open(REWARD_LOG_PATH, 'a') as f:  
    print('timesteps,reward,best_reward', file=f)

with open(X_POSITION_LOG_PATH, 'a') as f:  
    print('timesteps,x_position,best_x_position', file=f)


# Define TrainAndSaveCallback
class TrainAndSaveCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndSaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    # def _init_callback(self):
    #     os.makedirs(self.save_path, exist_ok=True)  # Ensure the folder for saving models exists

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:  # Checks whether the current number of steps is a multiple of check frequency that we set
            model_path = os.path.join(
                self.save_path, '1-2_02model_{}'.format(self.n_calls))  # Saves the model
            self.model.save(model_path)

            # Test and save each milestone's average reward and best reward
            total_reward = [0] * EPISODE_NUMBERS  # Record the total reward of each episode
            total_time = [0] * EPISODE_NUMBERS  # Record the current time step of each episode
            x_position = [0] * EPISODE_NUMBERS  # Record x position when each episode ends
            best_x_position = 0  # Record the right-most ending x position among all the episodes
            best_reward = 0  # Record the best reward among all the episodes

            for i in range(EPISODE_NUMBERS):
                state = env.reset()  # reset for each new trial
                done = False
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = model.predict(state)
                    state, reward, done, info = env.step(action)
                    x_position[i] = info['x_pos']
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                
                if x_position[i] > best_x_position:
                    best_x_position = x_position[i]

                state = env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', 10000000)
            print('average reward:', np.mean(total_reward),
                  'average time:', np.mean(total_time),
                  'best_reward:', best_reward)

            # Write into log CSV file
            with open(REWARD_LOG_PATH, 'a') as f:
                print(self.n_calls, ',', np.mean(total_reward), ',', best_reward, file=f)

            with open(X_POSITION_LOG_PATH, 'a') as f:
                print(self.n_calls, ',', np.mean(x_position), ',', best_x_position, file=f)

            # https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-more-values
            # todo: 
            # self.logger.record("random_value", value)

        return True



callback = TrainAndSaveCallback(check_freq=100000, save_path=DATA_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR,
            learning_rate=0.0001, n_steps=512, batch_size=64, n_epochs=10, gamma=0.9, gae_lambda=1.0, ent_coef=0.01)
model.learn(total_timesteps=10000000, callback=callback)

model.save('1-2-02model')

# TODO: count time
# TODO: set level and episode

#####################################################################
# Load the training result
#####################################################################
# model = PPO.load('./1-1-model')
# state = env.reset()
# while True:
#     action, _ = model.predict(state)
#     print(SIMPLE_MOVEMENT[action[0]])
#     state, reward, done, info = env.step(action)
#     env.render()


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



'''

'''