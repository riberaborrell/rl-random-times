import functools
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_random_times.po.models import ActorCriticModel
from rl_random_times.spg.replay_memories import ReplayMemoryAdvantage as Memory
from rl_random_times.utils.statistics import Statistics
from rl_random_times.utils.numeric import cumsum_numpy as cumsum, normalize_array
from rl_random_times.utils.path import load_data, save_data, save_model, load_model, get_ac_stoch_dir_path

class ActorCriticStochastic:
    def __init__(self, env, env_name, n_steps_lim, gamma=1.0, expectation_type='random-time',
                 estimate_z=True, n_layers=2, d_hidden_layer=32, batch_size=100, batch_size_z=100, 
                 mini_batch_size_type='constant', mini_batch_size=1, actor_lr=1e-2, critic_lr=1e-2,
                 n_grad_iterations=100, seed=None, policy_noise=1.0, optim_type='sgd',
                 norm_adv=True, train_vf_iters=10, clip_vf_loss=True, clip_coef=0.2, vf_coef=0.5):

        if isinstance(env.action_space, gym.spaces.Box):
            self.is_action_continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete) or \
             isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.is_action_continuous = False
            #TODO: check this case
            raise ValueError('Discrete action space case is not implemented yet')
        else:
            raise ValueError('Action space must be either continuous or discrete.')

        # agent name
        if self.is_action_continuous:
            self.agent = 'stoch-ac-cont-{}'.format(expectation_type)
        else:
            self.agent = 'stoch-ac-discrete-{}'.format(expectation_type)

        # environment and state and action dimension
        self.env_name = env_name
        self.env = env
        self.n_steps_lim = n_steps_lim

        # discount
        self.gamma = gamma

        # get state and action dimensions
        if 'EnvPool' in type(env).__name__ or hasattr(env.unwrapped, 'is_vectorized'):
            self.state_dim = env.observation_space.shape[0]
            if self.is_action_continuous:
                self.action_dim = env.action_space.shape[0]
            else:
                self.n_actions = env.action_space.n

        else:
            self.state_dim = env.observation_space.shape[1]
            if self.is_action_continuous:
                self.action_dim = env.action_space.shape[1]
            else:
                self.n_actions = env.action_space.n

        # expectation type
        self.expectation_type = expectation_type

        # on-policy expectation
        if expectation_type == 'on-policy':
            self.estimate_z = estimate_z
            self.mini_batch_size = mini_batch_size
            self.mini_batch_size_type = mini_batch_size_type

        # normalize advantages
        self.norm_adv = norm_adv

        # stochastic policy
        self.policy_noise = policy_noise
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer

        # initialize actor and critic models (policy and value function)
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        self.model = ActorCriticModel(
            self.state_dim, self.action_dim, hidden_sizes,
                activation=nn.Tanh(), std_init=policy_noise, seed=seed,
        )

        # stochastic gradient descent
        self.batch_size = batch_size
        if self.expectation_type == 'on-policy':
            batch_size_z = batch_size if batch_size_z is None else batch_size_z
            assert batch_size_z >= batch_size, 'The batch size z must be greater or equal to the batch size.'
            self.batch_size_z = batch_size_z
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.n_grad_iterations = n_grad_iterations

        # optimizer
        self.optim_type = optim_type
        if optim_type == 'adam':
            self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=critic_lr)
        elif optim_type == 'sgd':
            self.actor_optimizer = optim.SGD(self.model.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.SGD(self.model.critic.parameters(), lr=critic_lr)
        else:
            raise ValueError('The optimizer {optim} is not implemented')

        # value function loss
        self.train_vf_iters = train_vf_iters
        self.clip_vf_loss = clip_vf_loss
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef

        # seed
        self.seed = seed

    def sample_trajectories(self):

        K = self.batch_size if self.expectation_type == 'random-time' else self.batch_size_z

        # preallocate arrays
        initial_returns = np.zeros(K, dtype=np.float32)
        time_steps = np.empty(K, dtype=np.int32)

        # preallocate lists to store states, actions, rewards and values
        states, actions, rewards, values = [], [], [], []

        # reset environment

        # custom vectorized environment
        if hasattr(self.env.unwrapped, 'is_vectorized'):
            state, _ = self.env.reset(seed=self.seed, options={'batch_size': K})

        # gym vect env or envpool env
        else:
            state, _ = self.env.reset()

        # terminated and done flags
        been_terminated = np.full((K,), False)
        new_terminated = np.full((K,), False)
        done = np.full((K,), False)

        # sample which trajectories are going to be stored
        inds_mem = random.sample(range(K), self.batch_size)

        k = 1
        while not done.all():

            # sample action
            state_torch = torch.FloatTensor(state)
            action, _ = self.model.sample_action(state_torch)
            with torch.no_grad():
                value = self.model.get_value(state_torch)

            # save state, action and value
            states.append(state[inds_mem])
            actions.append(action[inds_mem])
            values.append(value[inds_mem])

            # step dynamics forward
            state, r, terminated, truncated, _ = self.env.step(action)

            # update terminated flags
            new_terminated = terminated & ~been_terminated
            been_terminated = terminated | been_terminated

            # done flags
            done = np.logical_or(been_terminated, truncated)

            # save reward
            rewards.append(r[inds_mem])

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
        trajs_states, trajs_actions, trajs_rewards, trajs_values, \
                trajs_returns, trajs_advantages = [], [], [], [], [], []

        # TODO: do we need this?
        #last_states = torch.tensor(
        #    np.vstack([states[n_final-1][i] for i, n_final in enumerate(time_steps)]),
        #    dtype=torch.float32,
        #)
        #with torch.no_grad():
        #    last_values = self.model.get_value(last_states).reshape(1, -1)

        # TODO: can this code be vectorized?
        for i in range(self.batch_size):
            idx_traj = inds_mem[i]
            n_final = time_steps[idx_traj]
            trajs_states.append(np.stack(states)[:n_final, i])
            trajs_actions.append(np.stack(actions)[:n_final, i])
            trajs_rewards.append(np.stack(rewards)[:n_final, i])
            trajs_values.append(np.stack(values)[:n_final, i].squeeze())

            # compute returns
            trajs_returns.append(cumsum(trajs_rewards[i]))

            # estimate advantages by computeing the temporal difference of the value function 
            deltas = np.empty(n_final, dtype=np.float32)
            for n in reversed(range(n_final)):
                if n == n_final - 1:
                    deltas[n] = trajs_rewards[i][n] - trajs_values[i][n]
                else:
                    deltas[n] = trajs_rewards[i][n] + self.gamma * trajs_values[i][n+1] - trajs_values[i][n]
            trajs_advantages.append(deltas)

        trajs_actions = np.vstack(trajs_actions) if self.is_action_continuous else np.hstack(trajs_actions)
        return np.vstack(trajs_states), trajs_actions, np.hstack(trajs_values), np.hstack(trajs_returns), \
               np.hstack(trajs_advantages), initial_returns, time_steps


    def sample_loss_random_time_trajectories(self, it):
        ''' Sample and compute alternative loss function corresponding to (random time) trajectory-based
            policy gradient. Compute gradients and update the policy parameters.
        '''

        # sample trajectories
        states, actions, values, returns, advantages, initial_returns, time_steps = self.sample_trajectories()

        # convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        values = torch.FloatTensor(values)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # normalize advantages
        if self.norm_adv:
            advantages = normalize_array(advantages, eps=1e-5)

        # compute log probs
        _, log_probs = self.model.actor.forward(states, actions)

        # pg loss
        phi = - log_probs * advantages
        pg_loss = phi.sum() / self.batch_size
        with torch.no_grad():
            pg_loss_var = phi.var().numpy()

        # reset gradients, compute gradients and update parameters
        self.actor_optimizer.zero_grad()
        pg_loss.backward()
        self.actor_optimizer.step()

        # value function loss
        for i in range(self.train_vf_iters):
            new_values = self.model.critic(states).view(-1)
            if self.clip_vf_loss:
                vf_loss_unclipped = (new_values - returns) ** 2
                vf_clipped = values + torch.clamp(
                    new_values - values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                vf_loss_clipped = (vf_clipped - returns) ** 2
                vf_loss_max = torch.max(vf_loss_unclipped, vf_loss_clipped)
                vf_loss = 0.5 * vf_loss_max.mean()
            else:
                vf_loss = 0.5 * ((new_values - returns) ** 2).mean()

            # reset gradients, compute gradients and update parameters
            self.critic_optimizer.zero_grad()
            vf_loss.backward()
            self.critic_optimizer.step()

        return pg_loss.detach().numpy(), pg_loss_var, vf_loss.detach().numpy(), initial_returns, time_steps

    def sample_loss_on_policy_state(self, it):
        ''' Sample and compute alternative loss function corresponding to (on-policy) state-space-based
            policy gradient. Compute gradients and update the policy parameters.
        '''

        # sample trajectories
        states, actions, values, returns, deltas, initial_returns, time_steps = self.sample_trajectories()

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
        memory.store_vectorized(states, actions, returns, deltas, values)

        # sample batch of experiences from memory
        if self.mini_batch_size_type == 'adaptive':
            K_mini = round(memory.size / self.mini_batch_size)
        else:
            K_mini = self.mini_batch_size
        batch = memory.sample_batch(K_mini)


        # compute log probs
        _, log_probs = self.model.actor.forward(batch['states'], batch['actions'])

        # estimate mean trajectory length
        mean_length = time_steps.mean() if self.estimate_z else 1

        # normalize advantages
        advantages = batch['advantages']
        if self.norm_adv:
            advantages = normalize_array(advantages, eps=1e-5)

        # pg loss
        phi = - (log_probs * advantages)
        pg_loss = phi.mean()
        with torch.no_grad():
            pg_loss_var = phi.var().numpy()

        # value function loss
        new_values = self.model.critic(batch['states']).view(-1)
        if self.clip_vf_loss:
            vf_loss_unclipped = (new_values - batch['returns']) ** 2
            vf_clipped = values + torch.clamp(
                new_values - batch['values'],
                -self.clip_coef,
                self.clip_coef,
            )
            vf_loss_clipped = (vf_clipped - batch['returns']) ** 2
            vf_loss_max = torch.max(vf_loss_unclipped, vf_loss_clipped)
            vf_loss = 0.5 * vf_loss_max.mean()
        else:
            vf_loss = 0.5 * ((new_values - batch['returns']) ** 2).mean()

        # loss
        loss = pg_loss + vf_loss * self.vf_coef

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

        return loss, pg_loss_var, initial_returns, time_steps

    def run_ac(self, backup_freq=None, live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        dir_path = get_ac_stoch_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(dir_path)

        # save algorithm parameters
        excluded = ['env', 'model', 'optimizer', 'live_lr']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, dir_path)

        # create object to store the is statistics of the learning
        stats = Statistics(
            eval_freq=1,
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
        save_model(self.model, dir_path, 'model_n-it{}'.format(0))

        if live_plot_freq:
            #TODO: live plot returns and time steps
            pass

        for i in np.arange(self.n_grad_iterations+1):

            # start timer
            ct_initial = time.time()

            # sample loss function
            if self.expectation_type == 'random-time':
                loss, loss_var, vf_loss, returns, time_steps = self.sample_loss_random_time_trajectories(i)
            else: #expectation_type == 'on-policy':
                loss, loss_var, returns, time_steps = self.sample_loss_on_policy_state(i)

            # end timer
            ct_final = time.time()

            # save and log epoch 
            stats.save_epoch(i, returns, time_steps, loss=loss,
                             loss_var=loss_var, ct=ct_final - ct_initial)
            stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models
            if backup_freq and (i + 1) % backup_freq== 0:
                save_model(self.model, dir_path, 'model_n-it{}'.format(i + 1))

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
