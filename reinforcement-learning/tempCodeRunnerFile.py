obs = None
done = True
# print(gym.__version__)
env.reset()
# print(obs.shape)

for step in range(1):
    obs, reward, terminated, info = env.step(
        env.action_space.sample())
    print(obs.shape)
    print(obs[0])
    plt.imshow(obs[0])
    plt.show()
    done = terminated
    if done:
        env.reset()
    env.render()
env.close()