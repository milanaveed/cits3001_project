from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import time

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
action = 0
counter = 0
for step in range(10000):

    with open('test_write.txt', 'a') as file:
        file.write("New line\n")
        file.write("Action:" + str(action) + "\n")
        

    time.sleep(0.0000000000000000000000000000001)
    
    if(action == 4):
        counter += 1
    #After 16 steps switch action
    if(action == 4 and counter > 16):
        action = 0
        counter = 0
    elif(action == 0):
        action = 4
    
    print(action)  
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        state = env.reset()
env.close()
