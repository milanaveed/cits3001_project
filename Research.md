# Project Research
## Slowing the simulator down
When running the project with the test code, I found that it would run very fast. This was apparently due to my CPU being really good. But in order to make good observations I needed to slow Mario down.
In order to do this I imported the time module and then during every step I used the sleep function for a very small value:
```
import time
...
for step in range(5000):
  ...
  time.sleep(0.0000000000001)
```
> Note: I added this for observation purposes, not for the final project submission.
---
## Recognising Enemies and Making Observations
As per [Link](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) it appears that the agent makes an observation of its environment in each step, returned by env.step() (the first tuple it returns).
Not yet sure how to use this data, but it seems like this is the right way to do it, as Lauren (Tuesday Lab Instructor) said that the way Mario recognises enemies is through these observations and the pixels that make up these enemies.
Apparently David suggested that when we do `env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")` we could use `'SuperMarioBros-v3'` which displays everything as single pixels, making it easier to understand?

Using `v3` means it is easier to get it working, but this would result in less marks. I asked and they said that getting the code to run in `v0` gets you bonus marks. They haven't disclosed how many marks that is yet.

Bonus marks will also be awarded if Mario can traverse multiple levels.

![Image](https://user-images.githubusercontent.com/2184469/40948817-3cd6600a-6830-11e8-8abb-9cee6a31d377.png)

## Mario's Moveset
If we take a look at the lines:
```
action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```
This is what determines Mario's next move. `.sample()` generates a number from 0-6, each of which indicates one of Mario's possible moves.
If we run the test code and replace `action` in `env.step(action)` with one of these numbers, we can observe how this changes what Mario will do.

`env.step(0)` -> Mario does nothing

`env.step(1)` -> Mario moves RIGHT

`env.step(2)` -> Mario JUMPS (once) + moves RIGHT

`env.step(3)` -> Mario SPRINTS RIGHT

`env.step(4)` -> Mario JUMPS (once) + SPRINTS RIGHT

`env.step(5)` -> Mario JUMPS and does nothing

`env.step(6)` -> Mario moves LEFT



## More sources to read into
[gymnasium.Env](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)

[Gymnasium.vector.VectorEnv](https://gymnasium.farama.org/api/vector/#observation_space)

[Graphical Glitch After Call to reset](https://github.com/Kautenja/gym-super-mario-bros/issues/72)

[numpy.ndarray Attributes](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

[Q-Netwrok Mario](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/)


## Reinforcement Learning
The unit coordinator mentioned algorithms from stable baselines for the project and I researched a bit about it.
<br/>

---------------------Below answered by ChatGPT-4---------------------

### What are Stable Baselines algorithms?
Stable Baselines is a set of high-quality implementations of reinforcement learning algorithms in Python using the TensorFlow library. Stable Baselines3 is an improvement over Stable Baselines and is based on PyTorch. The goal of these libraries is to provide reliable and efficient implementations of standard reinforcement learning algorithms. Here are some of the algorithms that are included in these libraries:

#### Stable Baselines (TensorFlow-based)
1. **Proximal Policy Optimization (PPO)** - A popular policy optimization method.
2. **Trust Region Policy Optimization (TRPO)** - Another policy optimization method which tends to be more sample-efficient compared to methods like DDPG.
3. **Deep Deterministic Policy Gradient (DDPG)** - An off-policy algorithm that can be used for continuous action spaces.
4. **Actor-Critic using Kronecker-Factored Trust Region (ACKTR)** - This is a more complex policy optimization method that makes use of more advanced optimization techniques.
5. **Twin Delayed DDPG (TD3)** - An extension of DDPG that introduces several new mechanisms to stabilize training.
6. **Soft Actor-Critic (SAC)** - An off-policy actor-critic deep RL algorithm that aims to maximize the expected reward while also maximizing entropy.
7. **A2C (Advantage Actor-Critic)** - A policy gradient method that leverages an advantage function to reduce variance in the policy updates.
8. **DQN (Deep Q-Networks)** - A value-based method that leverages deep neural networks to estimate Q-values.
   
#### Stable Baselines3 (PyTorch-based)
1. **PPO (Proximal Policy Optimization)** - The PyTorch implementation of PPO.
2. **A2C (Advantage Actor-Critic)** - The PyTorch implementation of A2C.
3. **DQN (Deep Q-Networks)** - The PyTorch implementation of DQN.
4. **SAC (Soft Actor-Critic)** - The PyTorch implementation of SAC.
5. **TD3 (Twin Delayed DDPG)** - The PyTorch implementation of TD3.

These libraries also contain utilities for recording and visualizing the results of reinforcement learning experiments, making it easier to analyze and debug the training process. It is highly recommended to consult the official documentation for the respective libraries (Stable Baselines and Stable Baselines3) for detailed information on each algorithm and how to use it in your reinforcement learning projects.

### Which Stable baselines algorithm is suitable for the Super Mario Bros project?
For a project where you are using a Reinforcement Learning (RL) approach to train an agent to play "Super Mario Bros," selecting an appropriate RL algorithm is crucial. Given your scenario where you're working with a discrete action space and a potentially complex state space, here are some algorithms that you might consider, and why they might be suitable:

1. **PPO (Proximal Policy Optimization)**: PPO is widely used in practice for a range of RL problems because of its stability and efficiency. It would be a good starting point for your project.

2. **DQN (Deep Q-Network)**: DQN could be another great choice, especially because it handles discrete action spaces well, which is the case for a game like "Super Mario Bros." DQN with extensions such as double DQN, prioritized replay, and dueling DQN can potentially enhance performance further.

3. **A2C (Advantage Actor-Critic)** or **A3C (Asynchronous Advantage Actor-Critic)**: These actor-critic methods could also be useful as they try to combine the benefits of value-based and policy-based methods.

Here is a general approach to setting up your project:

1. **Environment Setup**: Make sure to correctly set up the `gym-super-mario-bros` environment and familiarize yourself with the state space, action space, and rewards.
   
2. **Algorithm Selection**: Start with a simpler algorithm (like PPO or DQN) to establish a baseline performance.

3. **Feature Engineering**: Consider if any feature engineering is required to help the agent learn efficiently. Sometimes simplifying the state space or engineering certain features can make learning more efficient.

4. **Hyperparameter Tuning**: Once you have a baseline, experiment with different hyperparameters to optimize the performance of your chosen algorithm.

5. **Training**: Train your agent for a sufficient number of episodes to allow it to learn a good policy. Be prepared for this to potentially take a long time, depending on the complexity of your setup.

6. **Evaluation**: Regularly evaluate your agent's performance on different levels to ensure it is learning a general strategy, not just overfitting to particular levels.

I would recommend starting with implementing a well-documented algorithm like PPO from Stable Baselines3 library to get started with your project. Good luck!

### Where can we find good resources about PPO algorithm?
Absolutely, learning about Proximal Policy Optimization (PPO) can indeed be a significant asset, especially for your current project. Here are some resources where you can learn more about PPO:

#### Papers and Documentation

1. **[Original PPO Paper](https://arxiv.org/abs/1707.06347)**: Understanding PPO directly from the original paper can be quite beneficial, although it might be a bit technical.
   
2. **[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)**: OpenAI's educational resource provides a great explanation of PPO along with Python code.

3. **[Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)**: The documentation gives a detailed explanation of how to use the PPO implementation in the Stable Baselines3 library.

#### Blogs and Articles

4. **[Towards Data Science](https://towardsdatascience.com/)**: You can find several articles explaining PPO in a simplified manner on this platform. Just search for "Proximal Policy Optimization" in the search bar.

5. **[Medium](https://medium.com/)**: Similar to Towards Data Science, you can find various articles on PPO here. Search for PPO to find related articles.

#### Video Tutorials

6. **[YouTube](https://www.youtube.com/)**: There are many YouTubers who explain RL algorithms including PPO. Channels such as Sentdex and David Silver’s Reinforcement Learning Course might be beneficial.

#### GitHub and Communities

8. **[GitHub](https://github.com/)**: You can find various PPO implementations and projects on GitHub which can give you a practical understanding of how to work with PPO.

9. **[Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)**: Subreddits like this one can be good places to ask specific questions and get answers from experienced individuals in the field.

10. **[Stack Overflow](https://stackoverflow.com/)**: You can ask questions and find previous questions and answers about technical issues you might encounter while working with PPO.

Remember to complement these resources with hands-on practice by implementing PPO and experimenting with different configurations to see how it performs in different settings. Good luck!

----------------------Above answered by ChatGPT-4----------------------


