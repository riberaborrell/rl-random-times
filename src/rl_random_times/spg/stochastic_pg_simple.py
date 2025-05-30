from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_random_times.spg.models import CategoricalPolicy, GaussianPolicyLearntCov
from rl_random_times.utils.numeric import discounted_cumsum_list, normalize_array
from rl_random_times.utils.path import load_data, save_data, get_reinforce_simple_dir_path

class ReinforceStochastic:
    ''' Basic implementation of the REINFORCE algorithm i.e. stochastic policy gradient
        where the q-value function is approximated by the return at that time step.
    '''
    def __init__(self, env, gamma, n_layers, d_hidden_layer, lr, n_episodes, seed,
                 policy_type: Optional[str] = None, policy_noise: Optional[str] = None,
                 optim_type='adam'):

        # environment id
        self.env_id = 'bla'#env.spec.id

        if isinstance(env.action_space, gym.spaces.Box):
            self.is_action_continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.is_action_continuous = False
        else:
            raise ValueError('Action space must be either continuous or discrete.')

        # agent name
        if self.is_action_continuous:
            self.agent = 'reinforce-cont-simple'
        else:
            self.agent = 'reinforce-discrete-simple'

        # environment and state/action dimensions
        self.env = env
        state_dim = env.observation_space.shape[0]
        if self.is_action_continuous:
            action_dim = env.action_space.shape[0]
        else:
            n_actions = env.action_space.n

        # discount
        self.gamma = gamma

        # policy
        self.policy_type = policy_type
        self.policy_noise = policy_noise
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        if self.is_action_continuous:
            self.policy = GaussianPolicyLearntCov(
                state_dim, action_dim, hidden_sizes, activation=nn.Tanh(),
                std_init=policy_noise, seed=seed,
            )
        else:
            self.policy = CategoricalPolicy(state_dim, n_actions, hidden_sizes,
                                            activation=nn.Tanh(), seed=seed)

        # sgd
        self.optim_type = optim_type
        self.lr = lr
        if self.optim_type == 'adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        elif self.optim_type == 'sgd':
            self.optimizer = optim.SGD(self.policy.parameters(), lr=self.lr)
        else:
            raise ValueError('Optimizer must be either adam or sgd.')

        # number of episodes
        self.n_episodes = n_episodes

        # seed
        self.seed = seed

    def update_policy(self, rewards, log_probs):

        # compute discounted returns
        returns = discounted_cumsum_list(rewards, self.gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = normalize_array(returns, eps=1e-5)

        # compute loss
        #loss = []
        #for log_prob, Gt in zip(log_probs, n_returns):
        #    loss.append(-log_prob * Gt)
        #loss = torch.hstack(loss).sum()
        loss = (-torch.hstack(log_probs) * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy().item()

    def run_reinforce(self, log_freq=100, load=False):

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
            state, _ = self.env.reset(seed=self.seed) if ep == 0 else self.env.reset()
            log_probs, rewards = [], []
            done = False
            while not done:

                # sample action from policy
                state_torch = torch.FloatTensor(state)
                action, _ = self.policy.sample_action(state_torch, log_prob=False)

                if self.is_action_continuous:
                    action_torch = torch.FloatTensor(action)
                else:
                    action_torch = torch.FloatTensor([action.item()])

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
            time_steps[ep] = self.env.get_wrapper_attr('_elapsed_steps')
            if ep % log_freq == 0:
                print('Ep.: {:d}, return: {:4.2f}, time steps: {:.1f}'.format(
                    ep, returns[ep], time_steps[ep]
                ))

        data = {
            'returns': returns,
            'time_steps': time_steps,
            'losses': losses,
        }
        save_data(data, dir_path)
        return True, data

def get_stats_multiple_datas(datas):

    # get number of episodes
    n_episodes = datas[0]['returns'].shape[0]

    # preallocate arrays
    array_shape = (len(datas), n_episodes)
    returns = np.empty(array_shape)
    time_steps = np.empty(array_shape)

    # load evaluation
    for i, data in enumerate(datas):
        returns[i] = data['returns']
        time_steps[i] = data['time_steps']

    return returns, time_steps
