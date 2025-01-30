import functools
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_random_times.dpg.models import DeterministicPolicy
from rl_random_times.dpg.replay_memories import ReplayMemoryModelBasedDPG as Memory
from rl_random_times.utils.statistics import Statistics
from rl_random_times.utils.schedulers import simple_lr_schedule#, two_phase_lr_schedule
from rl_random_times.utils.numeric import dot_vect, cumsum_numpy as cumsum, normalize_array
from rl_random_times.utils.path import load_data, save_data, save_model, load_model, get_reinforce_det_dir_path

class ModelBasedDeterministicPG:
    def __init__(self, env, env_id, n_steps_lim, expectation_type, return_type, gamma,
                 n_layers, d_hidden_layer, batch_size, lr, n_grad_iterations, seed,
                 estimate_z=None, batch_size_z=None, mini_batch_size=None, mini_batch_size_type='constant', optim_type='adam',
                 scheduled_lr=False, lr_final=None):

        assert isinstance(env.action_space, gym.spaces.Box), 'Action space must be continuous.'

        # agent name
        self.agent = 'reinforce-det-{}'.format(expectation_type)

        # environment and state and action dimension
        self.env_id = env_id
        self.env = env
        self.n_steps_lim = n_steps_lim

        # get state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # expectation type and return type
        self.expectation_type = expectation_type
        self.return_type = return_type

        # discount
        self.gamma = gamma

        # deterministic policy
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer

        # initialize policy model
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        self.policy= DeterministicPolicy(state_dim=self.state_dim, action_dim=self.action_dim,
                                         hidden_sizes=hidden_sizes, activation=nn.Tanh())

        # stochastic gradient descent
        self.batch_size = batch_size
        if self.expectation_type == 'on-policy':
            batch_size_z = batch_size if batch_size_z is None else batch_size_z
            assert batch_size_z >= batch_size, 'The batch size z must be greater or equal to the batch size.'
            self.batch_size_z = batch_size_z
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

        # scheduler
        self.scheduled_lr = scheduled_lr
        if scheduled_lr:
            self.lr_final = lr_final
            lr_schedule = functools.partial(simple_lr_schedule, lr_init=lr,
                                            lr_final=lr_final, n_iter=n_grad_iterations+1)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda it: 1)

        # seed
        self.seed = seed

        # on-policy expectation
        if expectation_type == 'on-policy':
            self.estimate_z = estimate_z
            self.mini_batch_size = mini_batch_size
            self.mini_batch_size_type = mini_batch_size_type

    def sample_trajectories(self):

        K = self.batch_size if self.expectation_type == 'random-time' else self.batch_size_z

        # preallocate arrays
        initial_returns = np.zeros(K, dtype=np.float32)
        time_steps = np.empty(K, dtype=np.int32)

        # preallocate lists to store states, actions and rewards
        states, dbts, rewards = [], [], []

        # reset environment
        # envpool env
        if 'EnvPool' in type(self.env).__name__:
            state, _ = self.env.reset()

        # custom vectorized environment
        else:
            state, info = self.env.reset(seed=self.seed, options={'batch_size': K})

        # terminated and done flags
        been_terminated = np.full((K,), False)
        new_terminated = np.full((K,), False)
        done = np.full((K,), False)

        # sample which trajectories are going to be stored
        idx_mem = random.sample(range(K), self.batch_size)

        k = 1
        while not done.all():

            # sample action
            state_torch = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = self.policy(state_torch).numpy()

            # save state and action
            states.append(state[idx_mem])

            # step dynamics forward
            state, r, terminated, truncated, info = self.env.step(action)

            dbts.append(info['dbt'][idx_mem])

            # update terminated flags
            new_terminated = terminated & ~been_terminated
            been_terminated = terminated | been_terminated

            # done flags
            done = np.logical_or(been_terminated, truncated)

            # save reward
            rewards.append(r[idx_mem])

            # save initial returns
            initial_returns[~been_terminated | new_terminated] += r[~been_terminated | new_terminated]

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
        trajs_states, trajs_dbts, trajs_rewards, trajs_returns = [], [], [], []

        for i in range(self.batch_size):
            idx_traj = idx_mem[i]
            idx = time_steps[idx_traj]
            trajs_states.append(np.stack(states)[:idx-1, i])
            trajs_dbts.append(np.stack(dbts)[:idx-1, i])
            trajs_rewards.append(np.stack(rewards)[:idx, i])

            # compute initial returns
            if self.return_type == 'initial-return':
                trajs_returns.append(np.full(time_steps[idx_traj]-1, initial_returns[idx_traj]))

            # compute n-step returns
            else: # return_type == 'n-return'
                trajs_returns.append(cumsum(trajs_rewards[i])[1:])

        return np.vstack(trajs_states), np.vstack(trajs_dbts), \
               np.hstack(trajs_returns), initial_returns, time_steps


    def sample_loss_random_time_trajectories(self, it):
        ''' Sample and compute alternative loss function corresponding to the (random-time)
            trajectory-based policy gradient. Compute gradient and update the policy parameters.
        '''

        # sample trajectories
        states, dbts, returns, initial_returns, time_steps = self.sample_trajectories()

        # convert to torch tensors
        states = torch.FloatTensor(states)
        dbts = torch.FloatTensor(dbts)
        returns = torch.FloatTensor(returns)

        # compute actions following the policy
        actions = self.policy.forward(states)

        # normalize returns
        #returns = normalize_array(returns, eps=1e-5)

        # compute alternative objective terms

        # policy x \nabla_a r(s, a)
        a = 0.5 * torch.linalg.norm(actions, axis=1).pow(2) * self.env.unwrapped.dt

        # policy x brownian increments
        b = dot_vect(dbts, actions)

        # calculate loss
        phi = a - returns * b

        # loss and loss variance
        loss = phi.sum() / self.batch_size
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # get actual learning rate
        self.live_lr = self.optimizer.param_groups[0]['lr'] if it != 0 else self.lr

        # reset gradients, compute gradients and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.detach().numpy(), loss_var, initial_returns, time_steps


    def sample_loss_on_policy_state(self, it):
        ''' Sample and compute alternative loss function corresponding to the (on-policy)
            state-based policy gradient. Compute gradient and update the policy parameters.
        '''


        # sample trajectories
        states, dbts, returns, initial_returns, time_steps = self.sample_trajectories()

        # initialize memory
        memory = Memory(
            size=states.shape[0]+1,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

        # store experiences in memory
        memory.store_vectorized(states, dbts, returns=returns)

        # sample batch of experiences from memory
        if self.mini_batch_size_type == 'adaptive':
            K_mini = round(memory.size / self.mini_batch_size)
        else:
            K_mini = self.mini_batch_size
        batch = memory.sample_batch(K_mini)

        # compute actions following the policy
        actions = self.policy.forward(batch['states'])

        # estimate mean trajectory length
        mean_length = time_steps.mean() if self.estimate_z else 1

        # normalize returns
        returns = batch['returns']
        #returns = normalize_array(returns, eps=1e-5)

        # compute alternative objective terms
        # policy x \nabla_a r(s, a)
        a = 0.5 * torch.linalg.norm(actions, axis=1).pow(2) * self.env.unwrapped.dt

        # policy x brownian increments
        b = dot_vect(batch['dbts'], actions)

        # calculate loss
        phi = a - batch['returns'] * b
        loss = phi.mean()
        with torch.no_grad():
            loss_var = phi.var().numpy()

        # reset and compute actor gradients
        self.optimizer.zero_grad()
        loss.backward()

        # scale gradients before updating parameters
        if self.estimate_z:
            with torch.no_grad():
                for param in self.policy.parameters():
                    if param.grad is not None:
                        param.grad *= mean_length

        self.live_lr = self.optimizer.param_groups[0]['lr'] if it != 0 else self.lr

        #update parameters
        self.optimizer.step()

        # update learning rate
        self.scheduler.step()

        return loss, loss_var, initial_returns, time_steps

    def run_reinforce(self, backup_freq=None, live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        dir_path = get_reinforce_det_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(dir_path)

        # save algorithm parameters
        excluded = ['env', 'policy', 'optimizer', 'scheduler', 'live_lr']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, dir_path)

        # create object to store the is statistics of the learning
        stats = Statistics(
            eval_freq=1,
            eval_batch_size=self.batch_size,
            n_iterations=self.n_grad_iterations,
            iter_str='grad. it.:',
            policy_type='det',
            track_loss=True,
            track_ct=True,
            track_lr=True if self.scheduled_lr else False,
        )
        keys_chosen = [
            'mean_lengths', 'var_lengths', 'max_lengths',
            'mean_returns', 'var_returns',
            'losses', 'loss_vars',
            'cts',
        ]
        keys_chosen += ['lrs'] if self.scheduled_lr else []

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
                loss, loss_var, returns, time_steps = self.sample_loss_random_time_trajectories(i)
            else: #expectation_type == 'on-policy':
                loss, loss_var, returns, time_steps = self.sample_loss_on_policy_state(i)

            # end timer
            ct_final = time.time()

            # save and log epoch 
            lr = self.live_lr if self.scheduled_lr else None
            track_lr=True if self.scheduled_lr else False,
            stats.save_epoch(i, returns, time_steps, loss=loss,
                             loss_var=loss_var, ct=ct_final - ct_initial, lr=lr)
            stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models
            if backup_freq and (i + 1) % backup_freq== 0:
                save_model(self.policy, dir_path, 'policy_n-it{}'.format(i + 1))

            # backup statistics
            if (i + 1) % 100 == 0:
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

    def get_stats_multiple_datas(self, datas):

        # preallocate arrays
        array_shape = (len(datas), self.n_grad_iterations+1)
        objectives = np.empty(array_shape)
        mean_lengths = np.empty(array_shape)
        max_lengths = np.empty(array_shape)
        #total_lengths = np.empty(array_shape)

        # load evaluation
        for i, data in enumerate(datas):
            objectives[i] = data['mean_returns']
            mean_lengths[i] = data['mean_lengths']
            max_lengths[i] = data['max_lengths']
            #total_lengths[i] = data['total_lengths']

        return objectives, mean_lengths, max_lengths#, total_lengths

    def load_backup_model(self, data, i=0):
        try:
            load_model(self.policy, data['dir_path'], file_name='policy_n-it{}'.format(i))
            return True
        except FileNotFoundError as e:
            print('There is no backup for grad. iteration {:d}'.format(i))
            return False

    def get_means_and_stds(self, env, data, iterations):

        n_iterations = len(iterations)
        means = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        stds = np.empty((n_iterations, env.n_states, env.d), dtype=np.float32)
        for i, it in enumerate(iterations):
            load_backup_model(self, data, it)
            mean, std = evaluate_stoch_policy_model(env, data['policy'])
            means[i] = mean.reshape(env.n_states, env.d)
            stds[i] = std.reshape(env.n_states, env.d)
        return means, stds
        if expectation_type == 'on-policy' and mini_batch_size is None:
            raise ValueError('The mini_batch_size must be provided when using on-policy')
