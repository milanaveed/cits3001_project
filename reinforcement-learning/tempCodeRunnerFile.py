env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# # env = JoypadSpace(env, SIMPLE_MOVEMENT)
# # SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
# # sample the ['right'] and ['right', 'A'] actions
# MY_MOVEMENT = SIMPLE_MOVEMENT[1:3] 
# env = JoypadSpace(env, MY_MOVEMENT)
# print(env.observation_space.shape) # (240, 256, 3)
# # print(MY_MOVEMENT) # [['right'], ['right', 'A']]

# # Skip frames
# # env = SkipFrame(env, skip=4)

# # Grayscale environment
# env = GrayScaleObservation(env, keep_dim=True)
# print(env.observation_space.shape) # (240, 256, 1)

# # Resize observation
# env = ResizeObservation(env, shape=84)
# print(env.observation_space.shape) # (84, 84, 1)


# # Create a vectorized wrapper for an environment
# env = DummyVecEnv([lambda: env])

# # # Stack the four frames each time
# env = VecFrameStack(env, 4, channels_order='last')
# print(env.observation_space.shape) # (1, 84, 84, 4)