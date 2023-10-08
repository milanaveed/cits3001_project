# #import the game
import gym
import gym_super_mario_bros
import os
import signal # Ignore keyboard interruption and prevent accidental quits
# #import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# #import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# sample the ['right'] and ['right', 'A'] actions
MY_MOVEMENT = SIMPLE_MOVEMENT[1:3] 
env = JoypadSpace(env, MY_MOVEMENT)
print(env.observation_space.shape) # (240, 256, 3)

#########################################################################
#PREVENT ACCIDENTAL QUITS
#########################################################################
cur_pid=os.getpid()
print(cur_pid)

def SigIntHand(SIG, FRM):
    print("Ctrl-C does not work on the cmd prompt.")
    print("List the process id by `tasklist | findstr python`")
    print(f"taskkill /PID {cur_pid} /F")

signal.signal(signal.SIGINT, SigIntHand)

done = True
# Loop through each frame in the game
for step in range(10000):
    # Start the game to begin with
    if done:
        # Start the game
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()