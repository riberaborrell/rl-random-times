import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_for_gym.spg.models import GaussianPolicyLearntCov
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.path import get_reinforce_simple_dir_path, load_data, save_data, save_model, load_model
from rl_for_gym.utils.plots import plot_y_per_episode

class ReinforceStochastic:
    # REINFORCE with continuous actions
    def __init__(self, env, gamma, policy_type, policy_noise, n_layers, d_hidden_layer, lr, n_episodes,
                 seed, optim_type='adam'):
    #def __init__(self, env, gamma: float = 0.99, policy_type, policy_noise, lr: float= 1e-4, n_episodes: int = 1000, seed: int = 0):

        self.agent = 'reinforce-stoch-simple'

        # environment and state/action dimensions
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # discount
        self.gamma = gamma

        # policy
        self.policy_type = policy_type
        self.policy_noise = policy_noise
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        self.policy = GaussianPolicyLearntCov(state_dim, action_dim, hidden_sizes,
                                              activation=nn.Tanh(), std_init=policy_noise)

        # sgd
        self.optim_type = optim_type
        self.lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # number of episodes
        self.n_episodes = n_episodes

        # seed
        self.seed = seed

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

        return loss.detach().numpy().item()

    def run_reinforce(self, live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        dir_path = get_reinforce_simple_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(dir_path)

        # preallocate returns and time steps
        returns = np.empty(self.n_episodes)
        time_steps = np.empty(self.n_episodes)
        losses = np.empty(self.n_episodes)

        for ep in np.arange(self.n_episodes):
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
                    losses[ep] = self.update_policy(rewards, log_probs)
                    break

            # save return and time steps
            returns[ep] = sum(rewards)
            time_steps[ep] = self.env._elapsed_steps
            print('Ep.: {:d}, return: {:4.2f}, time steps: {:.1f}'.format(
                ep, returns[ep], time_steps[ep]
            ))

        data = {
            'returns': returns,
            'time_steps': time_steps,
        }
        save_data(data, dir_path)
        return True, data

def main():
    args = get_base_parser().parse_args()

    # restrict to continuous action spaces
    assert args.env_id in ["MountainCarContinuous-v0", "Pendulum-v1"], ''

    # create gym env 
    env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim)

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        env,
        gamma=args.gamma,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden,
        lr=args.lr,
        n_episodes=args.n_episodes,
        seed=args.seed,
        optim_type=args.optim_type,
    )

    # run reinforce with random time horizon 
    succ, data = agent.run_reinforce(
        log_freq=args.log_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    x = np.arange(args.n_episodes)
    plot_y_per_episode(x, data['returns'], run_window=100, title='Returns', legend=True)
    plot_y_per_episode(x, data['time_steps'], run_window=100, title='Time steps')


if __name__ == '__main__':
    main()
