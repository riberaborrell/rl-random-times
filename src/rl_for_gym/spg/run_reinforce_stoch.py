import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_for_gym.spg.models import GaussianPolicyLearntCov
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.path import load_data, save_data, get_dir_path


class ReinforceStoch:
    # REINFORCE with continuous actions
    def __init__(self, env, lr: float= 1e-4, gamma: float = 0.99):

        self.env = env
        self.d_state = env.observation_space.shape[0]
        self.d_action = env.action_space.shape[0]
        self.policy = GaussianPolicyLearntCov(self.d_state, self.d_action, hidden_sizes=[64, 64],
                                              activation=nn.Tanh(), std_init=1.)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-5
        )
        loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * Gt)
        self.optimizer.zero_grad()
        loss = torch.hstack(loss).sum()
        loss.backward()
        self.optimizer.step()

    def run_reinforce(self, n_episodes, seed, log_freq, load):

        # get dir path
        dir_path = get_dir_path(self.env, algorithm_name='reinforce_stoch')

        # load results
        if load:
            return load_data(dir_path)

        for ep in np.arange(n_episodes):
            state, _ = self.env.reset()
            log_probs, rewards = [], []
            done = False
            while not done:

                # sample action from policy
                state_torch = torch.FloatTensor(state)
                action, _ = self.policy.sample_action(state_torch, log_prob=False)
                action_torch = torch.FloatTensor(action)

                # step environment dynamics forward 
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # compute log prob and store rewards
                _, log_prob = self.policy.forward(state_torch, action_torch)
                log_probs.append(log_prob)
                rewards.append(reward)

                # update state
                state = next_state

                if done:
                    self.update_policy(rewards, log_probs)
                    print(f"Episode {ep}, Total Reward: {sum(rewards)}, Time steps: {self.env._elapsed_steps}")
                    break

def main():
    args = get_base_parser().parse_args()

    # restrict to continuous action spaces
    assert args.env_id in ["MountainCarContinuous-v0", "Pendulum-v1"], ''

    # create gym env 
    env = gym.make(args.env_id)

    # reinforce stochastic agent
    agent = ReinforceStoch(env, args.lr, args.gamma)

    # run reinforce with random time horizon 
    succ, data = agent.run_reinforce(
        n_episodes=args.n_episodes,
        seed=args.seed,
        log_freq=args.log_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return


if __name__ == '__main__':
    main()
