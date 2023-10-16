from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import time

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
action = 6
counter = 0

jump_ceiling = 2
backward_ceiling = 0
while jump_ceiling < 101:
    time_alive = 0
    for step in range(5000):
        time_alive += 1
        #time.sleep(0.000000000000000000000000000000000000001)
        
        counter += 1
        
        if(action == 4 and counter > jump_ceiling):
            action = 6
            counter = 0
        elif(action == 6 and counter > backward_ceiling):
            action = 4
            counter = 0
        
        #print(action)  
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            state = env.reset()
    #env.close()
    
    with open('test_write.txt', 'a') as file:
        file.write("Time alive:" + str(time_alive) + " Jump Ceiling:" + str(jump_ceiling) + " Backward Ceiling:" + str(backward_ceiling) + "\n")
    
    backward_ceiling += 1
    
    if backward_ceiling == jump_ceiling:
        jump_ceiling += 1
        backward_ceiling = 0
    env.reset()
env.close()
    
