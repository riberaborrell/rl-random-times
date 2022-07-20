import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from base_parser import get_base_parser

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

class Policy():
    '''
    '''
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.Tanh(),
            nn.Linear(32, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_prob_dist = self.network(torch.FloatTensor(state))
        return action_prob_dist

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def discount_cumsum2(x, gamma):
    n = len(x)
    y = np.array(
        [gamma**i * x[i] for i in np.arange(n)]
    )
    z = y[::-1].cumsum()[::-1]
    return z
    #return z - z.mean()

def normalize_advs_trick(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)

def reinforce(env_name='CartPole-v0', gamma=0.99, lr=0.01, n_episodes=2000,
              batch_size=10):

    # initialize environment 
    env = gym.make(env_name)

    # initialize policy
    model = Policy(env)
    #s = env.reset()
    #print(model.predict(s))
    #print(model.network(torch.FloatTensor(s)))

    # preallocate lists to hold results
    batch_states = []
    batch_actions = []
    batch_discounted_returns = []
    batch_counter = 0
    total_returns = []
    total_time_steps = []

    # define optimizer
    optimizer = optim.Adam(
        model.network.parameters(),
        lr=lr,
    )

    action_space = np.arange(env.action_space.n)
    for ep in np.arange(n_episodes):

        # reset state
        state = env.reset()

        # preallocate rewards for the episode
        ep_rewards = []

        # time step
        k = 0

        complete = False
        while complete == False:

            # save state
            batch_states.append(state.copy())

            # get action following policy
            action_prob_dist = model.predict(state).detach().numpy()
            action = np.random.choice(action_space, p=action_prob_dist)

            # next step
            state, r, complete, _ = env.step(action)
            k += 1

            # save action and reward
            batch_actions.append(action)
            ep_rewards.append(r)

        # update batch data
        batch_discounted_returns.extend(discount_cumsum(ep_rewards, gamma))
        batch_counter += 1
        total_returns.append(sum(ep_rewards))
        total_time_steps.append(k)

        # update network if batch is complete 
        if batch_counter == batch_size:

            # reset gradients ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            state_tensor = torch.FloatTensor(batch_states)
            action_tensor = torch.LongTensor(batch_actions)
            batch_discounted_returns = normalize_advs_trick(batch_discounted_returns)
            returns_tensor = torch.FloatTensor(batch_discounted_returns)

            # calculate loss
            log_action_prob_dists = torch.log(model.predict(state_tensor))
            log_probs = log_action_prob_dists[np.arange(len(action_tensor)), action_tensor]
            breakpoint()
            loss = - (returns_tensor * log_probs).mean()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_states = []
            batch_actions = []
            batch_discounted_returns = []
            batch_counter = 0

            # print running average
            run_avg_msg = 'ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}'.format(
                ep + 1,
                np.mean(total_returns[-batch_size:]),
                np.mean(total_time_steps[-batch_size:]),
            )
            print(run_avg_msg)

    return total_returns, total_time_steps, model

def main():
    args = get_parser().parse_args()

    # run reinforce
    returns, time_steps, model = reinforce(
        gamma=args.gamma,
        lr=args.lr,
        n_episodes=args.n_episodes_lim,
        batch_size=args.batch_size,
    )

    window = args.batch_size

    # plot returns
    smoothed_returns = [
        np.mean(returns[i-window:i+1]) if i > window
        else np.mean(returns[:i+1]) for i in range(len(returns))
    ]
    plt.figure(figsize=(12, 8))
    plt.plot(returns)
    plt.plot(smoothed_returns)
    plt.ylabel('Total Returns')
    plt.xlabel('Episodes')
    plt.show()

    # plot time steps
    smoothed_time_steps = [
        np.mean(time_steps[i-window:i+1]) if i > window
        else np.mean(time_steps[:i+1]) for i in range(len(time_steps))
    ]
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps)
    plt.plot(smoothed_time_steps)
    plt.ylabel('Total Time steps')
    plt.xlabel('Episodes')
    plt.show()


if __name__ == '__main__':
    main()
