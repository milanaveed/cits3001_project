# #import the game
import gym
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

# * Simplify the movements: use RIGHT_ONLY?

####################################################################
# Set up game environment and simplify actions
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# sample the ['right'] and ['right', 'A'] actions
MY_MOVEMENT = SIMPLE_MOVEMENT[1:3] 
env = JoypadSpace(env, MY_MOVEMENT)
print(env.observation_space.shape) # (240, 256, 3)
# print(MY_MOVEMENT) # [['right'], ['right', 'A']]

# Skip frames
# env = SkipFrame(env, skip=4)

# Grayscale environment
env = GrayScaleObservation(env, keep_dim=True)
print(env.observation_space.shape) # (240, 256, 1)

# Resize observation
env = ResizeObservation(env, shape=84)
print(env.observation_space.shape) # (84, 84, 1)


# Create a vectorized wrapper for an environment
env = DummyVecEnv([lambda: env])

# # Stack the four frames each time
env = VecFrameStack(env, 4, channels_order='last')
print(env.observation_space.shape) # (1, 84, 84, 4)

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


###############################################################
# Prevent accidental quits when training
###############################################################
cur_pid=os.getpid()
print(cur_pid)

def SigIntHand(SIG, FRM):
    print("Ctrl-C does not work on the cmd prompt.")
    print("List the process id by `tasklist | findstr python`")
    print("COMMAND THAT CAN KILL THE PROCESS:")
    print(f"taskkill /PID {cur_pid} /F")

signal.signal(signal.SIGINT, SigIntHand)


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
                self.save_path, '1-1_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


DATA_DIR = './reinforcement-learning/train/'
LOG_DIR = './reinforcement-learning/logs/'

callback = TrainAndSaveCallback(check_freq=500000, save_path=DATA_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR,
            learning_rate=0.000001, n_steps=512)
model.learn(4000000, callback=callback)

model.save('1-1-model')

# started the current training from 17:50 Sunday


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
