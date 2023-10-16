# #import the game
# import gym
import time
import gym_super_mario_bros
import os
import signal # Ignore keyboard interruption and prevent accidental quits
# #import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# #import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# # from wrappers import SkipFrame, 
from matplotlib import pyplot as plt


# Beginner Tutorial: https://www.youtube.com/watch?v=2eeYqJ0uBKE


start_time = time.time()

####################################################################
# Set up game environment and simplify actions
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# sample the ['right'] and ['right', 'A'] actions
# MY_MOVEMENT = SIMPLE_MOVEMENT[1:3] 
# env = JoypadSpace(env, MY_MOVEMENT)
print(env.observation_space.shape) # (240, 256, 3)
# print(MY_MOVEMENT) # [['right'], ['right', 'A']]

# Skip frames
# env = SkipFrame(env, skip=4)

# Grayscale environment
env = GrayScaleObservation(env, keep_dim=True)
print(env.observation_space.shape) # (240, 256, 1)

# Resize observation
# env = ResizeObservation(env, shape=84)
# print(env.observation_space.shape) # (84, 84, 1)


# Create a vectorized wrapper for an environment
env = DummyVecEnv([lambda: env])

# # Stack the four frames each time
env = VecFrameStack(env, 4, channels_order='last')
print(env.observation_space.shape) # (1, 84, 84, 4)


###############################################################
# Prevent accidental quits when training
###############################################################
# cur_pid=os.getpid()
# print(cur_pid)

# def SigIntHand(SIG, FRM):
#     print("Ctrl-C does not work on the cmd prompt.")
#     print("List the process id by `tasklist | findstr python`")
#     print("COMMAND THAT CAN KILL THE PROCESS:")
#     print(f"taskkill /PID {cur_pid} /F")

# signal.signal(signal.SIGINT, SigIntHand)


###############################################################
# Train and save the training process
###############################################################

DATA_DIR = './reinforcement-learning/train1-1-99/'
os.makedirs(DATA_DIR, exist_ok=True)
LOG_DIR = './reinforcement-learning/logs1-1-99/'
TRAINING_TIME_LOG_PATH = DATA_DIR + 'training_time_log.csv'

with open(TRAINING_TIME_LOG_PATH, 'a') as f:
    print('timesteps,early_version', file=f)

class TrainAndSaveCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndSaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            current_time = time.time()
            # Get time difference in minutes
            elapsed_time = round((current_time - start_time) / 60)

            model_path = os.path.join(
                self.save_path, '1-1_99model_{}'.format(self.n_calls))
            self.model.save(model_path)

            print('time steps:', self.n_calls, '/', 10000000)
            print('time taken for training (minutes):', elapsed_time)

            with open(TRAINING_TIME_LOG_PATH, 'a') as f:
                print(self.n_calls, ',', elapsed_time, file=f)

        return True



callback = TrainAndSaveCallback(check_freq=10000, save_path=DATA_DIR)
model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=LOG_DIR,
            learning_rate=0.000001, n_steps=512)
model.learn(100000, callback=callback)

model.save('1-1-99model')

# started the current training from 17:50 Sunday


#####################################################################
# Testing the game and show frames
#####################################################################
# state = env.reset()
# state, reward, done, info = env.step([env.action_space.sample()])
# state, reward, done, info = env.step([env.action_space.sample()])
# state, reward, done, info = env.step([env.action_space.sample()])
# plt.figure(figsize=(15,12))
# for idx in range(state.shape[3]):
#     plt.subplot(1,4,idx+1)
#     plt.imshow(state[0][:,:,idx])
# plt.show()


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
#     state, reward, done, info = env.step(env.action_space.sample())
#     # Show the game on the screen
#     env.render()
# # Close the game
# env.close()
