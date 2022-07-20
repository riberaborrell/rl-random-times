# script debugged from https://github.com/jachiam/rl-intro

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim

from base_parser import get_base_parser
from models import FeedForwardNN, TwoLayerNN

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

class ReplayBuffer:

    def __init__(self, obs_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

def dqn(env_name='CartPole-v0', hidden_dim=32, n_layers=2,
        lr=1e-3, gamma=0.99, n_epochs=50, steps_per_epoch=5000,
        batch_size=32, target_update_freq=2500, final_epsilon=0.05,
        finish_decay=50000, replay_size=25000, steps_before_training=100):

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    # initialize qvalue and target qvalue representations
    hidden_sizes = [hidden_dim for i in range(n_layers -1)]
    #model = FeedForwardNN(d_in=obs_dim, hidden_sizes=hidden_sizes, d_out=n_acts)
    #target_model = FeedForwardNN(d_in=obs_dim, hidden_sizes=hidden_sizes, d_out=n_acts)
    model = TwoLayerNN(d_in=obs_dim, hidden_size=32, d_out=n_acts)
    target_model = TwoLayerNN(d_in=obs_dim, hidden_size=32, d_out=n_acts)

    # set same parameters
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(param.data)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def get_action(model, obs, eps):
        if np.random.rand() < eps:
            return np.random.randint(n_acts)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs)
                phi = model.forward(obs_tensor)
                return torch.argmax(phi).item()

    def update_parameters(model, target_model, batch, gamma):
        obs = torch.tensor(batch['obs1'])
        obs_next = torch.tensor(batch['obs2'])
        act = torch.tensor(batch['acts'])
        rew = torch.tensor(batch['rews'])
        done = torch.tensor(batch['done'])

        # evaluate model at the current states
        phi = model.forward(obs)

        # q value for the given pairs of states and actions
        q_val = torch.where(act==0, phi[:, 0], phi[:, 1])

        # evaluate target model at next states
        target_phi = target_model.forward(obs_next)

        # get max q-value w.r.t the actions
        q_val_next = torch.max(target_phi, axis=1)[0]

        # compute y_i (using target networks)
        y_i = rew + gamma * q_val_next

        # Bellman error loss 
        mse_loss = nn.MSELoss()
        loss = mse_loss(q_val, y_i)

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().item()


    def test_q(n_test_eps=10):
        ep_rets, ep_lens = [], []
        for _ in range(n_test_eps):
            obs, rew, done, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(done):
                env.render()
                obs, rew, done, _ = test_env.step(get_action(model, obs, final_epsilon))
                ep_ret += rew
                ep_len += 1
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
        return np.mean(ep_rets), np.mean(ep_lens)

    obs, rew, done, epsilon, ep_ret, ep_len = env.reset(), 0, False, 1, 0, 0
    epoch_losses, epoch_rets, epoch_lens, epoch_qs = [], [], [], []
    total_steps = n_epochs * steps_per_epoch + steps_before_training
    for t in range(total_steps):
        act = get_action(model, obs, epsilon)
        next_obs, rew, done, _ = env.step(act)
        replay_buffer.store(obs, act, rew, next_obs, done)
        obs = next_obs
        ep_ret += rew
        ep_len += 1

        if done:
            epoch_rets.append(ep_ret)
            epoch_lens.append(ep_len)
            obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t > steps_before_training:
            batch = replay_buffer.sample_batch(batch_size)
            step_loss = update_parameters(model, target_model, batch, gamma)
            epoch_losses.append(step_loss)

        if t % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = 1 + (final_epsilon - 1)*min(1, t/finish_decay)

        # at the end of each epoch, evaluate the agent
        if (t - steps_before_training) % steps_per_epoch == 0 and (t - steps_before_training)>0:
            epoch = (t - steps_before_training) // steps_per_epoch
            test_ep_ret, test_ep_len = test_q()
            print(('epoch: %d \t loss: %.3f \t train_ret: %.3f' \
                   + '\t train_len: %.3f \t test_ret: %.3f \t test_len: %.3f ' \
                   + '\t mean q: %.3f \t epsilon: %.3f')%
                    (epoch, np.mean(epoch_losses), np.mean(epoch_rets),
                     np.mean(epoch_lens), test_ep_ret, test_ep_len,
                     np.mean(epoch_qs), epsilon))
            epoch_losses, epoch_rets, epoch_lens, epoch_qs = [], [], [], []
def main():
    args = get_parser().parse_args()

    # run dqn 
    returns, time_steps, model = dqn(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    breakpoint()
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
