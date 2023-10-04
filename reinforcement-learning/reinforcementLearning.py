# #import the game
import gym
import gym_super_mario_bros
import os
# #import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# #import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
# from wrappers import SkipFrame, ResizeObservation
from matplotlib import pyplot as plt


# Beginner Tutorial: https://www.youtube.com/watch?v=2eeYqJ0uBKE
# Level Up Tutorial: https://www.youtube.com/watch?v=PxoG0A2QoFs
# Inspiration: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

# * Simplify the movements: use RIGHT_ONLY?

####################################################################
# Set up game environment and simplify actions
env = gym_super_mario_bros.make("SuperMarioBros-v0")
# sample the ['right'] and ['right', 'A'] actions
MY_MOVEMENT = SIMPLE_MOVEMENT[1:3]
env = JoypadSpace(env, MY_MOVEMENT)
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Skip frames
# env = SkipFrame(env, skip=4)

# Grayscale environment
env = GrayScaleObservation(env, keep_dim=True)

# Resize observation
# env = ResizeObservation(env, shape=84)

# Create a vectorized wrapper for an environment
env = DummyVecEnv([lambda: env])

# Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

# todo: can we resize the frame? see what it'll look like after resize
# todo: can we skip some frames?

###################################################################
# Possible improvement
###################################################################
#      # 1. CREATE THE MARIO ENV
#      mario_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
#      # 2. SIMPLIFY THE CONTROLS
#      mario_env = JoypadSpace(mario_env, SIMPLE_MOVEMENT)
#     # 3. SKIP FRAMES AND TAKE ACTION EVERY N FRAMES
#     mario_env = SkipFrame(mario_env, skip=4)
#     # 4. TRANSFORM OBSERVATIONS INTO GRAYSCALE
#     mario_env = GrayScaleObservation(mario_env)
#     # 5. RESIZE OBSERVATIONS TO REDUCE DIMENSIONALITY
#     mario_env = ResizeObservation(mario_env, shape=84) 
#     # 6. NORMALIZE OBSERVATIONS
#     mario_env = TransformObservation(mario_env, f=lambda x: x / 255.)
#     # 7. STACK N FRAMES TO INTRODUCE TEMPORAL ASPECT
#     mario_env = FrameStack(mario_env, num_stack=4)

#     return mario_env

# mario_env  = add_wrapper_functionality()


###############################################################
# Train and save the training process
###############################################################
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
            model_path = os.path.join(
                self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


DATA_DIR = './reinforcement-learning/train/'
LOG_DIR = './reinforcement-learning/logs/'

callback = TrainAndSaveCallback(check_freq=50000, save_path=DATA_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR,
            learning_rate=0.000005, n_steps=512)
model.learn(4000000, callback=callback)

model.save('test-model')

# started the current training from 1 am

#TODO: upgrade stablebaselines3 version to v1.18.0

#####################################################################
# Load the training result
#####################################################################
# model = PPO.load('./reinforcement-learning/train/best_model_40000')
# state = env.reset()
# while True:
#     action, _ = model.predict(state)
#     print(MY_MOVEMENT[action[0]])
#     state, reward, done, info = env.step(action)
#     env.render()




#####################################################################
# Testing the game
#####################################################################
# state = env.reset()
# state, reward, done, info = env.step([1])
# state, reward, done, info = env.step([1])
# state, reward, done, info = env.step([1])
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
# for step in range(100000):
#     # Start the game to begin with
#     if done:
#         # Start the gamee
#         env.reset()
#     # Do random actions
#     state, reward, done, info = env.step(env.action_space.sample())
#     # Show the game on the screen
#     env.render()
# # Close the game
# env.close()
