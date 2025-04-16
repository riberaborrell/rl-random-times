# script adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_random_times.po.models import ActorCriticModel
from rl_random_times.utils.statistics import Statistics
from rl_random_times.utils.numeric import cumsum_numpy as cumsum, normalize_array
from rl_random_times.utils.path import load_data, save_data, save_model, load_model, get_episodic_ppo_dir_path

class PPO:
    def __init__(self, env, env_name, n_steps_lim, gamma=1.0, n_layers=2, d_hidden_layer=32,
                 policy_noise_init=1.0, optim_type='sgd', batch_size=100, n_iterations=1000,
                 lr=1e-2, n_mini_batches=32, update_epochs=10, max_grad_norm=0.5,
                 norm_adv=True, clip_vloss=True, clip_coef=0.2, ent_coef=0.,
                 vf_coef=0.5, target_kl=None, seed=None, cuda=False, torch_deterministic=True):

        # agent name
        self.agent = 'episodic-ppo'

        # environments
        self.env = env
        self.env_name = env_name
        self.n_steps_lim = n_steps_lim

        # get state and action dimensions
        self.state_dim = env.single_observation_space.shape[0]
        self.action_dim = env.single_action_space.shape[0]

        # discount
        self.gamma = gamma

        # cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.torch_deterministic = torch_deterministic

        # stochastic policy
        self.policy_noise_init = policy_noise_init
        self.n_layers = n_layers
        self.d_hidden_layer = d_hidden_layer

        # initialize ppo agent. actor and critic networks
        hidden_sizes = [d_hidden_layer for i in range(n_layers -1)]
        self.model = ActorCriticModel(
            self.state_dim, self.action_dim, hidden_sizes,
                activation=nn.Tanh(), std_init=policy_noise_init, seed=seed,
        ).to(self.device)

        # stochastic gradient descent
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.n_mini_batches = n_mini_batches
        self.lr = lr
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm

        # optimizer
        self.optim_type = optim_type
        if self.optim_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        elif self.optim_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError('The optimizer {optim} is not implemented')

        # normalize advantages flag
        self.norm_adv = norm_adv

        # value function
        self.clip_vloss = clip_vloss

        # clipping
        self.clip_coef = clip_coef

        # loss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # kl divergence
        self.target_kl = target_kl

        # seed 
        self.seed = seed

    def sample_trajectories(self):

        #TODO: adapt for batch_size_z
        K = self.batch_size

        # preallocate arrays
        initial_returns = np.zeros(K, dtype=np.float32)
        time_steps = np.empty(K, dtype=np.int32)

        # preallocate lists to store states, actions, rewards and values
        states, actions, log_probs, rewards, values = [], [], [], [], []

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
            action, log_prob = self.model.sample_action(state_torch, log_prob=True)
            with torch.no_grad():
                value = self.model.get_value(state_torch)

            # save state, action and value
            states.append(state[inds_mem])
            actions.append(action[inds_mem])
            log_probs.append(log_prob[inds_mem])
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

        # compute returns and advantages
        trajs_states, trajs_actions, trajs_log_probs, trajs_rewards, trajs_values, \
                trajs_returns, trajs_advantages = [], [], [], [], [], [], []

        # TODO: do we need this?
        """
        last_states = torch.tensor(
            np.vstack([states[n_final-1][i] for i, n_final in enumerate(time_steps)]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            last_values = self.model.get_value(last_states).reshape(1, -1)
        """

        # TODO: can this code be vectorized?
        for i in range(self.batch_size):
            idx_traj = inds_mem[i]
            n_final = time_steps[idx_traj]
            trajs_states.append(np.stack(states)[:n_final, i])
            trajs_actions.append(np.stack(actions)[:n_final, i])
            trajs_log_probs.append(np.stack(log_probs)[:n_final, i])
            trajs_rewards.append(np.stack(rewards)[:n_final, i])
            trajs_values.append(np.stack(values)[:n_final, i].squeeze())

            # compute returns
            trajs_returns.append(cumsum(trajs_rewards[i]))

            # estimate advantages by computeing the temporal difference of the value function 
            deltas = np.empty(n_final, dtype=np.float32)
            for n in reversed(range(n_final)):
                if n == n_final - 1:
                    #deltas[n] = trajs_rewards[i][n] - values[n][idx_traj]
                    deltas[n] = trajs_rewards[i][n] - trajs_values[i][n]
                else:
                    #deltas[n] = trajs_rewards[i][n] + self.gamma * values[n+1][idx_traj]  - values[n][idx_traj]
                    deltas[n] = trajs_rewards[i][n] + self.gamma * trajs_values[i][n+1] - trajs_values[i][n]
            trajs_advantages.append(deltas)

        return np.vstack(trajs_states), np.vstack(trajs_actions), np.hstack(trajs_log_probs), \
               np.hstack(trajs_values), np.hstack(trajs_returns), \
               np.hstack(trajs_advantages), initial_returns, time_steps

    def run_ppo(self, backup_freq=None, live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        dir_path = get_episodic_ppo_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(dir_path)

        # save algorithm parameters
        excluded = ['env', 'model', 'optimizer', 'device']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, dir_path)

        # create object to store the is statistics of the learning
        stats = Statistics(
            eval_freq=1,
            n_iterations=self.n_iterations,
            iter_str='Iterations:',
            policy_type='stoch',
            track_pg_updates=True,
            track_ct=True,
        )
        keys_chosen = [
            'mean_lengths', 'var_lengths', 'max_lengths',
            'mean_returns', 'var_returns',
            'n_grad_updates', 'cts',
        ]

        # save model initial parameters
        save_model(self.model, dir_path, 'model_n-it{}'.format(0))

        for i in range(self.n_iterations):

            # start timer
            ct_initial = time.time()

            # sample trajectories
            states, actions, log_probs, values, returns, deltas, initial_returns, time_steps = self.sample_trajectories()

            # convert to torch tensors
            states = torch.Tensor(states).to(self.device)
            actions = torch.Tensor(actions).to(self.device)
            log_probs = torch.Tensor(log_probs).to(self.device)
            values = torch.Tensor(values).to(self.device)
            returns = torch.Tensor(returns).to(self.device)
            deltas = torch.Tensor(deltas).to(self.device)

            # n total steps in batch
            n_total_steps = states.shape[0]
            mini_batch_size = n_total_steps // self.n_mini_batches
            last_mini_batch_size = n_total_steps % self.n_mini_batches

            # Optimizing the policy and value network

            # batch indices
            b_inds = np.arange(n_total_steps, dtype=np.int32)
            clip_fracs = []

            # count policy gradient updates in the iteration
            n_pg_updates = 0

            for epoch in range(self.update_epochs):

                # shuffle data
                np.random.shuffle(b_inds)

                for j in range(0, self.n_mini_batches):

                    # get mini batch indices
                    start = j * mini_batch_size
                    end = start + mini_batch_size if j < self.n_mini_batches - 1 else n_total_steps
                    mb_inds = b_inds[start:end]

                    # compute new log probs and entropy of the updated gaussian distribution
                    new_values = self.model.critic(states[mb_inds])
                    dist, new_log_probs = self.model.actor(states[mb_inds], actions[mb_inds])
                    entropy = dist.entropy().sum(1)

                    # compute the ratio of the new and old log probs
                    log_ratio = new_log_probs - log_probs[mb_inds]
                    ratio = log_ratio.exp()

                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = deltas[mb_inds]
                    if self.norm_adv:
                        mb_advantages = normalize_array(mb_advantages, eps=1e-8)

                    # policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    new_values = new_values.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (new_values - returns[mb_inds]) ** 2
                        v_clipped = values[mb_inds] + torch.clamp(
                            new_values - values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - returns[mb_inds]) ** 2).mean()

                    # entropy loss (entropy bonus for exploration)
                    entropy_loss = entropy.mean()

                    # total loss
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    n_pg_updates += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            # end timer
            ct_final = time.time()

            #TODO: do we need this statistics?
            y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # save and log epoch 
            stats.save_epoch(i, initial_returns, time_steps, n_pg_updates=n_pg_updates, ct=ct_final - ct_initial)
            stats.log_epoch(i) if i % log_freq == 0 else None

            # backup models
            if backup_freq and (i + 1) % backup_freq== 0:
                save_model(self.model, dir_path, 'model_n-it{}'.format(i + 1))

            # backup statistics
            if (i + 1) % 100 == 0:
                stats_dict = {key: stats.__dict__[key] for key in keys_chosen}
                save_data(data | stats_dict, dir_path)

        stats_dict = {key: value for key, value in vars(stats).items() if key in keys_chosen}
        data = data | stats_dict
        save_data(data, dir_path)
        return True, data

    def load_backup_model(self, data, i=0):
        try:
            load_model(self.model, data['dir_path'], file_name='model_n-it{}'.format(i))
            return True
        except FileNotFoundError as e:
            print('There is no backup for iteration {:d}'.format(i))
            return False

