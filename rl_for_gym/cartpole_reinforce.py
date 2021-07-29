import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

from base_parser import get_base_parser

import torch
import torch.nn as nn
import torch.optim as optim

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
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_prob_dist = self.network(torch.FloatTensor(state))
        return action_prob_dist

def discount_rewards(rewards, gamma=0.99):
    k_last = len(rewards)
    r = np.array(
        [gamma**i * rewards[i] for i in np.arange(k_last)]
    )
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def reinforce(env, Policy, num_episodes=2000,
              batch_size=10, gamma=0.99):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(
        Policy.network.parameters(),
        lr=0.01,
    )

    action_space = np.arange(env.action_space.n)
    for ep in np.arange(num_episodes):
        # reset state
        state = env.reset()

        # preallocate trajectory
        states = []
        rewards = []
        actions = []

        complete = False
        while complete == False:
            # get action following policy
            action_prob_dist = Policy.predict(state).detach().numpy()
            action = np.random.choice(action_space, p=action_prob_dist)

            # save state and action
            states.append(state)
            actions.append(action)

            # next step
            state, r, complete, _ = env.step(action)

            # save reward 
            rewards.append(r)

        # batch data
        batch_rewards.extend(discount_rewards(rewards, gamma))
        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_counter += 1
        total_rewards.append(sum(rewards))

        # update network if batch is complete 
        if batch_counter == batch_size:
            # reset ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            state_tensor = torch.FloatTensor(batch_states)
            action_tensor = torch.LongTensor(batch_actions)
            reward_tensor = torch.FloatTensor(batch_rewards)

            # calculate loss
            log_action_prob_dists = torch.log(Policy.predict(state_tensor))
            log_probs = log_action_prob_dists[np.arange(len(action_tensor)), action_tensor]
            loss = - (reward_tensor * log_probs).mean()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_rewards = []
            batch_actions = []
            batch_states = []
            batch_counter = 1

        # print running average
        run_avg_msg = '\rEp: {} Average of last 10: {:.2f}'.format(
            ep + 1,
            np.mean(total_rewards[-10:]),
        )
        print(run_avg_msg, end="")

    return total_rewards

def main():
    args = get_parser().parse_args()

    # tested with with CartPole-v0
    env = gym.make(args.env_id)
    s = env.reset()

    pe = Policy(env)
    print(pe.predict(s))
    print(pe.network(torch.FloatTensor(s)))

    rewards = reinforce(env, pe)
    window = 10
    smoothed_rewards = [
        np.mean(rewards[i-window:i+1]) if i > window
        else np.mean(rewards[:i+1]) for i in range(len(rewards))
    ]

    plt.figure(figsize=(12, 8))
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show()

if __name__ == '__main__':
    main()
