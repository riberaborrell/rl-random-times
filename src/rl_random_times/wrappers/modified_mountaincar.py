"""
Wrapper that modifies the MountainCar Continuous environment such that the agent is motivated
to reach the top of the mountain quickly.
"""

import time

import numpy as np
import gymnasium as gym


#[docs]

class ModifiedMountainCarEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, running_cost: float = 1., terminal_reward: float = 0.):

        # pass environment without time limit wrapper
        super().__init__(env.env)

        # running cost and terminal reward
        self.running_cost = running_cost
        self.terminal_reward = terminal_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        """Steps through the environment. ."""

        # environment step
        (next_obs, reward, terminated, truncated, info) = self.unwrapped.step(action)

        # modify the reward
        reward = reward - self.running_cost if not terminated else self.terminal_reward

        done = terminated or truncated

        return (next_obs, reward, terminated, truncated, info)
