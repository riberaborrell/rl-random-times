from agent import Agent
from figures import MyFigure

import numpy as np

class BlackjackAgent(Agent):
    '''
    '''

    def reset_trajectory(self):
        self.states = np.empty((0, 3), dtype=np.int32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0)

    def save_history(self, state, action, r):
        self.states = np.vstack((self.states, np.array(state)))
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, r)

    def get_state(self, observation):
        state = observation
        if observation[2]:
            state = observation[:2] + (1,)
        else:
            state = observation[:2] + (0,)
        return state

    def mc_learning(self, n_steps_lim, alpha):

        # preallocate information for all epochs
        self.preallocate_episodes()

        # initialize n and q-values tables
        n_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
            self.env.observation_space[2].n,
            self.env.action_space.n,
        ), dtype=np.int32)

        q_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
            self.env.observation_space[2].n,
            self.env.action_space.n,
        ), dtype=np.float64)

        # episodes
        for ep in np.arange(self.n_episodes):

            # reset environment
            observation = self.env.reset()
            state = self.get_state(observation)

            # reset trajectory
            self.reset_trajectory()

            # terminal state flag
            complete = False

            # sample episode
            for k in np.arange(n_steps_lim):

                # interrupt if we are in a terminal state
                if complete:
                    break

                # pick greedy action (exploitation)
                epsilon = self.epsilons[ep]
                if np.random.rand() > epsilon:
                    action = np.argmax(q_table[state])

                # pick random action (exploration)
                else:
                    action = self.env.action_space.sample()

                # step dynamics forward
                new_observation, r, complete, _ = self.env.step(action)
                new_state = self.get_state(new_observation)

                # save state, actions and reward
                self.save_history(state, action, r)

                # update state
                state = new_state

            # compute return
            self.compute_discounted_rewards()
            self.compute_returns()

            # update q values
            n_steps_trajectory = self.states.shape[0]
            for k in np.arange(n_steps_trajectory):

                # state-action index
                state = self.states[k]
                action = self.actions[k]
                idx = tuple(state) + (action,)
                g = self.returns[k]

                n_table[idx] += 1
                #alpha = 1 / n_table[idx]
                q_table[idx] = q_table[idx] \
                             + alpha * (g - q_table[idx])

            # save time steps
            self.save_episode(ep, k)

            # logs
            if self.logs:
                msg = self.log_episodes(ep)
                print(msg)


        # save number of episodes
        self.n_episodes = ep + 1

        # update npz dict
        self.update_npz_dict_agent()

        # save q_values table
        self.last_q_table = q_table
        self.last_n_table = n_table
        self.npz_dict['last_q_table'] = q_table
        self.npz_dict['last_n_table'] = n_table


    def plot_sample_returns(self):
        y = np.vstack((self.sample_returns, self.avg_sample_returns))
        fig = MyFigure(self.dir_path, 'sample_returns')
        fig.plot_multiple_lines(self.episodes, y)

    def plot_total_rewards(self):
        fig = MyFigure(self.dir_path, 'total_rewards')
        y = np.vstack((self.total_rewards, self.avg_total_rewards))
        fig.plot_multiple_lines(self.episodes, y)

    def plot_time_steps(self):
        fig = MyFigure(self.dir_path, 'time_steps')
        fig.plot_one_line(self.episodes, self.time_steps)

    def plot_epsilons(self):
        fig = MyFigure(self.dir_path, 'epsilons')
        fig.set_plot_type('semilogy')
        fig.plot_one_line(self.episodes, self.epsilons)

    def plot_frequency(self):
        frequency_ua = np.sum(self.last_n_table[:, :, 1, :], axis=2)
        frequency_nua = np.sum(self.last_n_table[:, :, 0, :], axis=2)

        fig = MyFigure(self.dir_path, 'frequency_usable_ace')
        fig.axes[0].imshow(frequency_ua, origin='lower')
        fig.savefig(fig.file_path)

        fig = MyFigure(self.dir_path, 'frequency_not_usable_ace')
        fig.axes[0].imshow(frequency_nua, origin='lower')
        fig.savefig(fig.file_path)

    def plot_policy(self):
        stick_hit_table_ua = np.argmax(self.last_q_table[:, :, 1, :], axis=2)
        stick_hit_table_nua = np.argmax(self.last_q_table[:, :, 0, :], axis=2)

        fig = MyFigure(self.dir_path, 'policy_usable_ace')
        fig.axes[0].imshow(stick_hit_table_ua, origin='lower')
        fig.savefig(fig.file_path)

        fig = MyFigure(self.dir_path, 'policy_not_usable_ace')
        fig.axes[0].imshow(stick_hit_table_nua, origin='lower')
        fig.savefig(fig.file_path)



