"""
Wrapper that converts the Mujoco Reacher environment to an episodic environment.
"""

import time

import numpy as np
import gymnasium as gym


#[docs]

class EpisodicReacherEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, threshold_dist: float = 0.05,
                 reward_dist_weight: float = 0., reward_ctrl_weight: float = 0.1,
                 running_cost: float = 1., terminal_reward: float = 0.):

        # pass environment without time limit wrapper
        super().__init__(env.env)

        # threshold distance between the fingertip of the reacher and the target.
        self.threshold_dist = threshold_dist

        # reward parameters
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

        # running cost and terminal reward
        self.running_cost = running_cost
        self.terminal_reward = terminal_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def distance_to_target(self, obs):
        return np.linalg.norm(obs[8:10])

    def step(self, action):
        """Steps through the environment. ."""

        # environment step
        (next_obs, _, _, truncated, info) = self.unwrapped.step(action)

        # compute reward with customized reward arguments
        reward = (
            self.reward_dist_weight * info["reward_dist"]
            + self.reward_ctrl_weight * info["reward_ctrl"]
        )

        # compute distance to the target
        dist = self.distance_to_target(next_obs)

        # terminate if the distance to the target is less than the threshold
        terminated = dist < self.threshold_dist

        # modify the reward
        reward = reward - self.running_cost if not terminated else reward + self.terminal_reward

        done = terminated or truncated

        return (next_obs, reward, terminated, truncated, info)

class EpisodicReacherVectEnv(gym.vector.VectorWrapper):
    def __init__(self, env: gym.vector.VectorEnv, threshold_dist: float = 0.02,
                 reward_dist_weight: float = 0., reward_ctrl_weight: float = 0.1,
                 running_cost: float = 1., terminal_reward: float = 0.):

        # pass environment without time limit wrapper
        super().__init__(env)

        # threshold distance between the fingertip of the reacher and the target.
        self.threshold_dist = threshold_dist

        # reward parameters
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

        # running cost and terminal reward
        self.running_cost = running_cost
        self.terminal_reward = terminal_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def distance_to_target(self, obs):
        return np.linalg.norm(obs[:, 8:10], axis=1)

    def step(self, action):
        """Steps through the environment. ."""

        # environment step
        (next_obs, _, _, truncated, info) = self.unwrapped.step(action)

        # compute reward with customized reward arguments
        reward = (
            self.reward_dist_weight * info["reward_dist"]
            + self.reward_ctrl_weight * info["reward_ctrl"]
        )

        # compute distance to the target
        dist = self.distance_to_target(next_obs)

        # terminate if the distance to the target is less than the threshold
        terminated = dist < self.threshold_dist

        # modify the reward
        reward[~terminated] -= self.running_cost
        reward[terminated] += self.terminal_reward

        done = np.logical_or(terminated, truncated)

        return (next_obs, reward, terminated, truncated, info)
