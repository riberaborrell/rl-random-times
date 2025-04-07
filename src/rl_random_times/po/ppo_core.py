# script adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_random_times.po.models import ActorCriticModel
from rl_random_times.utils.statistics import Statistics
from rl_random_times.utils.numeric import normalize_array
from rl_random_times.utils.path import load_data, save_data, save_model, load_model, get_ppo_dir_path

class PPO:
    def __init__(self, env, env_name, n_envs, n_steps_lim, gamma, n_total_steps, n_layers=2,
                 d_hidden_layer=32, policy_noise_init=1.0, optim_type='sgd',
                 lr=3e-4, anneal_lr=True, n_mini_batches=32, update_epochs=10, max_grad_norm=0.5,
                 norm_adv=True, gae_lambda=0.95, clip_vloss=True, clip_coef=0.2, ent_coef=0.,
                 vf_coef=0.5, target_kl=None, seed=None, cuda=False, torch_deterministic=True):

        # agent name
        self.agent = 'ppo'

        # environments
        self.env = env
        self.env_name = env_name
        self.n_envs = n_envs
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
        self.batch_size = int(self.n_envs * self.n_steps_lim)
        self.n_mini_batches = n_mini_batches
        self.mini_batch_size = int(self.batch_size // self.n_mini_batches)
        self.n_total_steps = n_total_steps
        self.n_iterations = self.n_total_steps // self.batch_size
        self.lr = lr
        self.anneal_lr = anneal_lr
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

        # critic
        self.gae_lambda = gae_lambda

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



    def run_ppo(self, backup_freq=None, live_plot_freq=None, log_freq=100, load=False):

        # get dir path
        dir_path = get_ppo_dir_path(**self.__dict__)

        # load results
        if load:
            return load_data(dir_path)

        # save algorithm parameters
        excluded = ['env', 'model', 'optimizer']
        data = {key: value for key, value in vars(self).items() if key not in excluded}
        save_data(data, dir_path)

         # TRY NOT TO MODIFY: seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # ALGO Logic: Storage setup
        obs = torch.zeros((self.n_steps_lim, self.n_envs) + self.env.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.n_steps_lim, self.n_envs) + self.env.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.n_steps_lim, self.n_envs)).to(self.device)
        rewards = torch.zeros((self.n_steps_lim, self.n_envs)).to(self.device)
        dones = torch.zeros((self.n_steps_lim, self.n_envs)).to(self.device)
        values = torch.zeros((self.n_steps_lim, self.n_envs)).to(self.device)

        # create object to store the is statistics of the learning
        stats = Statistics(
            eval_freq=1,
            n_iterations=self.n_iterations,
            iter_str='Iterations:',
            policy_type='stoch',
            track_ct=True,
        )
        keys_chosen = [
            'mean_lengths', 'var_lengths', 'max_lengths',
            'mean_returns', 'var_returns',
            'cts',
        ]

        # save model initial parameters
        save_model(self.model, dir_path, 'model_n-it{}'.format(0))

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.n_envs).to(self.device)

        for iteration in range(self.n_iterations):

            # start timer
            ct_initial = time.time()

            # returns and lengths for each iteration
            ep_returns, ep_time_steps = [], []

            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - iteration / self.n_iterations
                lrnow = frac * self.lr
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.n_steps_lim):
                global_step += self.n_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, value = self.model.get_action_and_value(next_obs)
                    values[step] = torch.Tensor(value.flatten()).to(self.device)
                actions[step] = torch.Tensor(action).to(self.device)
                logprobs[step] = torch.Tensor(logprob).to(self.device)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.env.step(action)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                #print(step, terminations, truncations)
                #print(self.env.envs[0].env.env.env.env.env.env.env.env.distance_to_target(next_obs))
                if "episode" in infos:
                    for i in range(infos["_episode"].shape[0]):
                        if infos["_episode"][i] :
                            ep_returns.append(infos["episode"]["r"][i])
                            ep_time_steps.append(infos["episode"]["l"][i])

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.model.critic.forward(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.n_steps_lim)):
                    if t == self.n_steps_lim - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values


            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network

            # batch indices
            b_inds = np.arange(self.batch_size)
            clipfracs = []

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    mb_inds = b_inds[start:end]

                    newvalue = self.model.critic(b_obs[mb_inds])
                    dist, newlogprob = self.model.actor(b_obs[mb_inds], b_actions[mb_inds])
                    entropy = dist.entropy().sum(1)

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = normalize_array(mb_advantages, eps=1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            # end timer
            ct_final = time.time()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # save and log epoch 
            stats.save_epoch(iteration, np.array(ep_returns, dtype=np.float32),
                             np.array(ep_time_steps, dtype=np.float32), ct=ct_final - ct_initial)
            stats.log_epoch(iteration)# if i % log_freq == 0 else None

            # backup models
            if backup_freq and (iteration + 1) % backup_freq== 0:
                save_model(self.model, dir_path, 'model_n-it{}'.format(iteration + 1))

            # backup statistics
            if (iteration + 1) % 100 == 0:
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

