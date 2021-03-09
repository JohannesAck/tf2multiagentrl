[![Build Status](https://travis-ci.com/JohannesAck/tf2multiagentrl.svg?branch=master)](https://travis-ci.com/JohannesAck/tf2multiagentrl)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f047411b259245f28c60095cc8e44250)](https://app.codacy.com/gh/JohannesAck/tf2multiagentrl?utm_source=github.com&utm_medium=referral&utm_content=JohannesAck/tf2multiagentrl&utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/caf7f96ddc1d4bc6871d3999bfc63ecc)](https://www.codacy.com/gh/JohannesAck/tf2multiagentrl/dashboard?utm_source=github.com&utm_medium=referral&utm_content=JohannesAck/tf2multiagentrl&utm_campaign=Badge_Coverage)

# TensorFlow 2 Implementation of Multi-Agent Reinforcement Learning Approaches 

This repository contains a modular TF2 implementations of multi-agent versions of the RL methods DDPG 
([MADDPG](https://arxiv.org/abs/1706.02275)),
 TD3 ([MATD3](https://arxiv.org/abs/1910.01465)),
 [SAC](https://arxiv.org/abs/1801.01290) (MASAC) and
 [D4PG](https://arxiv.org/abs/1804.08617) (MAD4PG).
 It also implements [prioritized experience replay](https://arxiv.org/abs/1511.05952).
 
 In our experiments we found MATD3 to work the best and did not see find a benefit by using Soft-Actor-Critic
 or the distributional D4PG. However, it is possible that these methods may be benefitial in more
 complex environments, while our evaluation here focussed on the 
 [multiagent-particle-envs by openai](https://github.com/openai/multiagent-particle-envs).

## Code Structure
We provide the code for the agents in tf2marl/agents and a finished training loop with logging
powered by sacred in train.py.

We denote lists of variables corresponding to each agent with the suffix `_n`, i.e.
`state_n` contains a list of n state batches, one for each agent. 

## Useage

Use `python >= 3.6` and install the requirement with
```
pip install -r requirements.txt
```
Start an experiment with 
```
python train.py
```
As we use [sacred](https://github.com/IDSIA/sacred) for configuration management and logging, 
the configuration can be updated with their CLI, i.e.
```
python train.py with scenario_name='simple_spread' num_units=128 num_episodes=10000
```
and experiments are automatically logged to `results/sacred/`, or optionally also to a MongoDB.
To observe this database we recommend to use [Omniboard](https://github.com/vivekratnavel/omniboard).

 
## Acknowledgement
The environments in `/tf2marl/multiagent` are from [multiagent-particle-envs by openai](https://github.com/openai/multiagent-particle-envs)
with the exception of `inversion.py` and `maximizeA2.py`, which I added for debugging purposes.

The implementation of the segment tree used for prioritized replay is based on 
[stable-baselines](https://github.com/hill-a/stable-baselines)

