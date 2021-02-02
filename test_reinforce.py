from agent import Agent
from base_parser import get_base_parser

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import gym

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


def reinforce(env, agent, Policy, lr=0.01, num_episodes=2000,
              batch_size=10):

    # start agent batch of trajectories
    agent.reset_batch()

    # define optimizer
    optimizer = optim.Adam(
        Policy.network.parameters(),
        lr=lr,
    )

    action_space = np.arange(env.action_space.n)
    for ep in np.arange(num_episodes):
        # start agent trajectory
        agent.reset_trajectory()

        # reset state
        state = env.reset()

        complete = False
        while complete == False:
            # get action following policy
            action_prob_dist = Policy.predict(state).detach().numpy()
            action = np.random.choice(action_space, p=action_prob_dist)

            # save state and action
            agent.states = np.vstack((agent.states, state))
            agent.actions = np.append(agent.actions, action)

            # next step
            state, r, complete, _ = env.step(action)

            # save reward 
            agent.rewards = np.append(agent.rewards, r)

        # compute returns 
        agent.get_discounted_rewards_and_returns()

        # update batch data
        agent.update_batch()

        # update network if batch is complete 
        if agent.batch_traj_num == batch_size:
            # reset ..
            optimizer.zero_grad()

            # tensor states, actions and rewards
            state_tensor = torch.FloatTensor(agent.batch_states)
            action_tensor = torch.LongTensor(agent.batch_actions)
            avg_returns_tensor = torch.FloatTensor(agent.batch_avg_returns)

            # calculate loss
            log_action_prob_dists = torch.log(Policy.predict(state_tensor))
            log_probs = log_action_prob_dists[np.arange(len(action_tensor)), action_tensor]
            loss = - (avg_returns_tensor * log_probs).mean()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            agent.reset_batch()

        # print running average
        run_avg_msg = '\rEp: {} Runninga avg of last 10: {:.2f}'.format(
            ep + 1,
            np.mean(agent.total_rewards[-10:]),
        )
        print(run_avg_msg, end="")

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make(args.env_id)

    # create agent
    agent = Agent(env, args.gamma)

    # get policy
    pe = Policy(env)

    # start reinforce
    reinforce(env, agent, pe, args.lr)

if __name__ == '__main__':
    main()
