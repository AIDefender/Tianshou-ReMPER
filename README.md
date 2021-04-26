**Tianshou** ([天授](https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88)) is a reinforcement learning platform based on pure PyTorch. Unlike existing reinforcement learning libraries, which are mainly based on TensorFlow, have many nested classes, unfriendly API, or slow-speed, Tianshou provides a fast-speed modularized framework and pythonic API for building the deep reinforcement learning agent with the least number of lines of code.

I add several modifications to Tianshou to facilitate my research:
- Enable recording forward and backward timestep in the replay buffer
- Enable using ram as input for atari envs(performace not tested)
- Other minor changes for convenient training and logging