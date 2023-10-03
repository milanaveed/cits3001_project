# #import the game
import gym
import gym_super_mario_bros
# #import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# #import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

# Beginner Tutorial: https://www.youtube.com/watch?v=2eeYqJ0uBKE
# Level Up Tutorial: https://www.youtube.com/watch?v=PxoG0A2QoFs
# Inspiration: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

# * Simplify the movements: use RIGHT_ONLY?

####################################################################

# Set up game environment and simplify actions
env = gym_super_mario_bros.make("SuperMarioBros-v0")
MY_MOVEMENT = SIMPLE_MOVEMENT[1:3] # sample the ['right'] and ['right', 'A'] actions
env = JoypadSpace(env, MY_MOVEMENT)
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Grayscale environment
env = GrayScaleObservation(env, keep_dim=True)

# Create a vectorized wrapper for an environment
env = DummyVecEnv([lambda: env])

# Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

#todo: can we resize the frame?
#todo: can we skip some frames?

'''
     # 1. CREATE THE MARIO ENV
     mario_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
     # 2. SIMPLIFY THE CONTROLS
     mario_env = JoypadSpace(mario_env, SIMPLE_MOVEMENT)
    # 3. SKIP FRAMES AND TAKE ACTION EVERY N FRAMES
    mario_env = SkipFrame(mario_env, skip=4)
    # 4. TRANSFORM OBSERVATIONS INTO GRAYSCALE
    mario_env = GrayScaleObservation(mario_env)
    # 5. RESIZE OBSERVATIONS TO REDUCE DIMENSIONALITY
    mario_env = ResizeObservation(mario_env, shape=84) 
    # 6. NORMALIZE OBSERVATIONS
    mario_env = TransformObservation(mario_env, f=lambda x: x / 255.)
    # 7. STACK N FRAMES TO INTRODUCE TEMPORAL ASPECT
    mario_env = FrameStack(mario_env, num_stack=4)

    return mario_env

mario_env  = add_wrapper_functionality()
'''

# Testing the game
state = env.reset()
state, reward, done, info = env.step([1])
state, reward, done, info = env.step([1])
state, reward, done, info = env.step([1])
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
