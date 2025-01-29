"""
Wrapper that converts the Mujoco Swimmer environment to an episodic environment.
"""

import time

import numpy as np
import gymnasium as gym


#[docs]

class EpisodicSwimmerEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, threshold_fwd_dist: float = 3,
                 forward_reward_weight: float = 0., ctrl_cost_weight: float = 0.1,
                 running_cost: float = 1., terminal_reward: float = 0.):

        # pass environment without time limit wrapper
        super().__init__(env.env)

        # threshold forwad distance for the x-position
        self.threshold_fwd_dist = threshold_fwd_dist

        # reward parameters
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight

        # running cost and terminal reward
        self.running_cost = running_cost
        self.terminal_reward = terminal_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_x_position = info["x_position"]
        return obs, info

    def step(self, action):
        """Steps through the environment. ."""

        # environment step
        (next_obs, _, _, truncated, info) = self.unwrapped.step(action)

        # compute reward with customized reward arguments
        reward = (
            self.forward_reward_weight * info["reward_forward"]
            + self.ctrl_cost_weight * info["reward_ctrl"]
        )

        # terminate if it has moved forward in the x-position enough
        fwd_dist = info["x_position"] - self.initial_x_position
        terminated = fwd_dist >= self.threshold_fwd_dist

        # modify the reward
        reward = reward - self.running_cost if not terminated else reward + self.terminal_reward

        done = terminated or truncated

        return (next_obs, reward, terminated, truncated, info)
