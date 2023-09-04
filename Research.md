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

Bonuse marks will also be awarded if Mario can traverse multiple levels.

![Image](https://user-images.githubusercontent.com/2184469/40948817-3cd6600a-6830-11e8-8abb-9cee6a31d377.png)

## More sources to read into
[gymnasium.Env](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)

[Gymnasium.vector.VectorEnv](https://gymnasium.farama.org/api/vector/#observation_space)

[Graphical Glitch After Call to reset](https://github.com/Kautenja/gym-super-mario-bros/issues/72)

[numpy.ndarray Attributes](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

[Q-Netwrok Mario](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/)



