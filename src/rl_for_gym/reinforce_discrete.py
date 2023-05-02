import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from rl_for_gym.utils_numeric import discount_cumsum

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

    def forward(self, state):
        state = torch.FloatTensor(state)
        action_prob_dist = self.network(state)
        return action_prob_dist

def render_test_episode(env, model):

    # create gym env 
    render_env = gym.make(env.unwrapped.spec.id, render_mode='human')

    # possible actions
    action_space = np.arange(render_env.action_space.n)

    # reset environment
    obs, _ = render_env.reset()

    # terminal state flag
    done = False

    # truncated flag
    truncated = False

    while not done and not truncated:

        # get action following policy
        action_prob_dist = model.forward(obs).detach().numpy()
        action = np.random.choice(action_space, p=action_prob_dist)
        action = render_env.action_space.sample()

        # step dynamics forward
        obs, _, done, truncated, _ = render_env.step(action)

    render_env.close()

def reinforce(env, gamma=0.99, lr=0.01, n_episodes=2000,
              batch_size=10, seed=None, render=False):

    # initialize policy
    model = Policy(env)
    _, _ = env.reset(seed=seed)
    torch.manual_seed(seed)

    # preallocate lists to hold results
    batch_states = []
    batch_actions = []
    batch_discounted_returns = []
    batch_time_steps = np.empty(0, dtype=np.int32)
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
        obs, info = env.reset()

        # preallocate rewards for the episode
        ep_rewards = []

        # time step
        k = 0

        done = False
        truncated = False
        while not done and not truncated:

            # save state
            batch_states.append(obs.copy())

            # get action following policy
            action_prob_dist = model.forward(obs).detach().numpy()
            action = np.random.choice(action_space, p=action_prob_dist)

            # next step
            obs, r, done, truncated, info = env.step(action)
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
            returns_tensor = torch.FloatTensor(batch_discounted_returns)

            # calculate loss
            log_action_prob_dists = torch.log(model.forward(state_tensor))
            log_probs = log_action_prob_dists[np.arange(len(action_tensor)), action_tensor]
            loss = - (returns_tensor * log_probs).mean()
            #loss = - fhts * (returns_tensor * log_probs).mean()
            #loss = - (batch_fht_prob * returns_tensor * log_probs).mean()

            # calculate gradients
            loss.backward()

            # update coefficients
            optimizer.step()

            # reset batch
            batch_states = []
            batch_actions = []
            batch_discounted_returns = []
            batch_time_steps = np.empty(0, dtype=np.int32)
            batch_prob = np.zeros(1000)
            batch_counter = 0

            running_total_returns = np.mean(total_returns[-batch_size:])
            running_total_time_steps = np.mean(total_time_steps[-batch_size:])

            # print running average
            run_avg_msg = 'ep: {}, run avg returns: {:.2f}, run avg time steps: {:.2f}'.format(
                ep + 1,
                running_total_returns,
                running_total_time_steps,
            )
            print(run_avg_msg)

            if running_total_returns > env.spec.reward_threshold:
                print("Solved! Agent properly trained!")
                break

    # render episode
    if render:
        render_test_episode(env, model)

    return total_returns, total_time_steps, model
