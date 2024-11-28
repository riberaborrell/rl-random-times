import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_for_gym.spg.models import CategoricalPolicy, GaussianPolicyConstantCov, GaussianPolicyLearntCov
from rl_for_gym.spg.replay_memories import ReplayMemoryReturn as Memory
from rl_for_gym.utils.statistics import Statistics
from rl_for_gym.utils.numeric import cumsum_numpy as cumsum, normalize_array
from rl_for_gym.utils.path import load_data, save_data, save_model, load_model, get_reinforce_stoch_dir_path

class ReinforceStochastic:
    def __init__(self, env, env_id, expectation_type, return_type, gamma, n_layers, d_hidden_layer,
                 batch_size, lr, n_grad_iterations, seed, policy_type=None, policy_noise=None,
                 estimate_z=None, mini_batch_size=None, mini_batch_size_type='constant',
                 memory_size=int(1e6), optim_type='adam'):

        if isinstance(env.action_space, gym.spaces.Box):
            self.is_action_continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete) or \
             isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.is_action_continuous = False
        else:
            raise ValueError('Action space must be either continuous or discrete.')

        # agent name
        if self.is_action_continuous:
            self.agent = 'reinforce-cont-{}'.format(expectation_type)
        else:
            self.agent = 'reinforce-discrete-{}'.format(expectation_type)

        # environment and state and action dimension
        self.env_id = env_id
        self.env = env

        # get state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        if self.is_action_continuous:
            self.action_dim = env.action_space.shape[0]
        else:
            self.n_actions = env.action_space.n

        # expectation type and return type
        self.expectation_type = expectation_type
        self.return_type = return_type

        # discount
        self.gamma = gamma

        # stochastic policy
        self.policy_type = policy_type
        self.policy_noise = policy_noise
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer

        # initialize policy model
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        if self.is_action_continuous and policy_type == 'const-cov':
            self.policy = GaussianPolicyConstantCov(
                self.state_dim, self.action_dim, hidden_sizes,
                activation=nn.Tanh(), std=self.policy_noise, seed=seed,
            )
        elif self.is_action_continuous and policy_type == 'learnt-cov':
            self.policy = GaussianPolicyLearntCov(
                self.state_dim, self.action_dim, hidden_sizes,
                activation=nn.Tanh(), std_init=self.policy_noise, seed=seed,
            )
        else:
            self.policy = CategoricalPolicy(self.state_dim, self.n_actions, hidden_sizes,
                                            activation=nn.Tanh(), seed=seed)

        # stochastic gradient descent
        self.batch_size = batch_size
        self.lr = lr
        self.n_grad_iterations = n_grad_iterations

        # optimizer
        self.optim_type = optim_type
        if optim_type == 'adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        elif optim_type == 'sgd':
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        else:
            raise ValueError('The optimizer {optim} is not implemented')


        # seed
        self.seed = seed

        # on-policy expectation
        if expectation_type == 'on-policy':
            self.estimate_z = estimate_z
            self.mini_batch_size = mini_batch_size
            self.mini_batch_size_type = mini_batch_size_type
            self.memory_size = memory_size

    def sample_trajectories(self):

        # preallocate arrays
        initial_returns = np.zeros(self.batch_size, dtype=np.float32)
        time_steps = np.empty(self.batch_size, dtype=np.int32)

        # preallocate lists to store states, actions and rewards
        states, actions, rewards = [], [], []

        # reset environment
        state, _ = self.env.reset()

        # terminated and done flags
        been_terminated = np.full((self.batch_size,), False)
        new_terminated = np.full((self.batch_size,), False)
        done = np.full((self.batch_size,), False)

        k = 1
        while not done.all():

            # sample action
            state_torch = torch.FloatTensor(state)
            action, _ = self.policy.sample_action(state_torch)

            # save state and action
            states.append(state)
            actions.append(action)

            # step dynamics forward
            state, r, terminated, truncated, _ = self.env.step(action)

            # update terminated flags
            new_terminated = terminated & ~been_terminated
            been_terminated = terminated | been_terminated

            # done flags
            done = np.logical_or(been_terminated, truncated)

            # save reward
            rewards.append(r)

            # save time steps
            if new_terminated.any():
                time_steps[new_terminated] = k

            # interrupt if all trajectories have reached a terminal state or have been truncated 
            if done.all():
                time_steps[~been_terminated] = k
                break

            # increment time step
            k += 1

        # compute returns
        trajs_states, trajs_actions, trajs_rewards, trajs_returns = [], [], [], []

        for i in range(self.batch_size):
            idx = time_steps[i]
            trajs_states.append(np.stack(states)[:idx, i])
            trajs_actions.append(np.stack(actions)[:idx, i])
            trajs_rewards.append(np.stack(rewards)[:idx, i])
            initial_returns[i] = np.sum(trajs_rewards[i])

            # compute initial returns
            if self.return_type == 'initial-return':
                trajs_returns.append(np.full(time_steps[i], initial_returns[i]))

            # compute n-step returns
            else: # return_type == 'n-return'
                trajs_returns.append(cumsum(trajs_rewards[i]))

        trajs_actions = np.vstack(trajs_actions) if self.is_action_continuous else np.hstack(trajs_actions)
        return np.vstack(trajs_states), trajs_actions, \
               np.hstack(trajs_returns), initial_returns, time_steps


    def sample_loss_random_time(self):
        ''' Sample and compute loss function corresponding to the policy gradient with
            random time expectation. Also update the policy parameters.
        '''

        # sample trajectories
        states, actions, returns, initial_returns, time_steps = self.sample_trajectories()

        # convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        returns = torch.FloatTensor(returns)

        # normalize returns
        returns = normalize_array(returns, eps=1e-5)

        # compute log probs
        _, log_probs = self.policy(states, actions)

        # calculate loss
        phi = - log_probs * returns

        # loss and loss variance
        loss = phi.sum() / self.batch_size
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset gradients, compute gradients and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy(), loss_var, initial_returns, time_steps

    def sample_loss_on_policy(self):

        # sample trajectories
        states, actions, returns, initial_returns, time_steps = self.sample_trajectories()

        # initialize memory
        if  self.is_action_continuous:
            memory = Memory(
                size=states.shape[0]+1,
                state_dim=self.state_dim,
                action_dim=self.action_dim
            )
        else:
            memory = Memory(
                size=states.shape[0]+1,
                state_dim=self.state_dim,
                is_action_continuous=False,
            )

        # store experiences in memory
        memory.store_vectorized(states, actions, returns=returns)

        # sample batch of experiences from memory
        if self.mini_batch_size_type == 'adaptive':
            K_mini = round(memory.size / self.mini_batch_size)
        else:
            K_mini = self.mini_batch_size
        batch = memory.sample_batch(K_mini)
        _, log_probs = self.policy.forward(batch['states'], batch['actions'])

        # estimate mean trajectory length
        mean_length = time_steps.mean() if self.estimate_z else 1

        # normalize returns
        returns = batch['returns']
        returns = normalize_array(returns, eps=1e-5)

        # calculate loss
        phi = - (log_probs * returns)
        loss = phi.mean()
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset and compute actor gradients
        self.optimizer.zero_grad()
        loss.backward()

        # scale learning rate
        self.optimizer.param_groups[0]['lr'] *= mean_length

        #update parameters
        self.optimizer.step()

        # re-scale learning rate back
        self.optimizer.param_groups[0]['lr'] /= mean_length

        return loss, loss_var, initial_returns, time_steps

    def run_reinforce(self, backup_freq=None, live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        dir_path = get_reinforce_stoch_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(dir_path)

        # save algorithm parameters
        excluded = ['env', 'policy', 'optimizer']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, dir_path)

        # create object to store the is statistics of the learning
        stats = Statistics(
            eval_freq=1,
            eval_batch_size=self.batch_size,
            n_iterations=self.n_grad_iterations,
            iter_str='grad. it.:',
            policy_type='stoch',
            track_loss=True,
            track_ct=True,
        )
        keys_chosen = [
            'mean_lengths', 'var_lengths', 'max_lengths',
            'mean_returns', 'var_returns',
            'losses', 'loss_vars',
            'cts',
        ]

        # save model initial parameters
        save_model(self.policy, dir_path, 'policy_n-it{}'.format(0))

        if live_plot_freq:
            #TODO: live plot returns and time steps
            pass

        for i in np.arange(self.n_grad_iterations+1):

            # start timer
            ct_initial = time.time()

            # sample loss function
            if self.expectation_type == 'random-time':
                loss, loss_var, returns, time_steps = self.sample_loss_random_time()
            else: #expectation_type == 'on-policy':
                loss, loss_var, returns, time_steps = self.sample_loss_on_policy()

            # end timer
            ct_final = time.time()

            # save and log epoch 
            stats.save_epoch(i, returns, time_steps, loss=loss,
                             loss_var=loss_var, ct=ct_final - ct_initial)
            stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models and results
            if backup_freq and (i + 1) % backup_freq== 0:
                save_model(self.policy, dir_path, 'policy_n-it{}'.format(i + 1))
                stats_dict = {key: stats.__dict__[key] for key in keys_chosen}
                save_data(data | stats_dict, dir_path)

            # update plots
            if live_plot_freq and i % live_plot_freq == 0:
                #TODO: live plot returns and time steps
                pass

        stats_dict = {key: value for key, value in vars(stats).items() if key in keys_chosen}
        data = data | stats_dict
        save_data(data, dir_path)
        return True, data


    def load_backup_model(data, i=0):
        try:
            load_model(data['policy'], data['dir_path'], file_name='policy_n-it{}'.format(i))
            return True
        except FileNotFoundError as e:
            print('There is no backup for grad. iteration {:d}'.format(i))
            return False

    def get_means_and_stds(env, data, iterations):

        n_iterations = len(iterations)
        means = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        stds = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        for i, it in enumerate(iterations):
            load_backup_model(data, it)
            mean, std = evaluate_stoch_policy_model(env, data['policy'])
            means[i] = mean.reshape(env.n_states, env.d)
            stds[i] = std.reshape(env.n_states, env.d)
        return means, stds
        if expectation_type == 'on-policy' and mini_batch_size is None:
            raise ValueError('The mini_batch_size must be provided when using on-policy')



