from agent import Agent
from figures import MyFigure

import numpy as np

class WindyGridworldAgent(Agent):
    '''
    '''

    def reset_trajectory(self):
        self.states = np.empty((0, 2), dtype=np.int32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0)

    def save_history(self, state, action, r):
        self.states = np.vstack((self.states, np.array(state)))
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, r)

    def initialize_frequency_table(self):
        self.n_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
            self.env.action_space.n,
        ), dtype=np.int32)

    def initialize_q_table(self):
        self.q_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
            self.env.action_space.n,
        ), dtype=np.float64)

    def initialize_eligibility_traces(self):
        self.e_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
            self.env.action_space.n,
        ), dtype=np.float64)

    def get_epsilon_greedy_action(self, ep, state):
        # pick greedy action (exploitation)
        epsilon = self.epsilons[ep]
        if np.random.rand() > epsilon:
            action = np.argmax(self.q_table[state])

        # pick random action (exploration)
        else:
            action = self.env.action_space.sample()
        return action

    def mc_learning(self, n_episodes_lim, n_steps_lim):

        # preallocate information for all epochs
        self.n_episodes = n_episodes_lim
        self.preallocate_episodes()

        # initialize frequency and q-values table
        self.initialize_frequency_table()
        self.initialize_q_table()

        # set epsilons
        self.set_glie_epsilons()

        # episodes
        for ep in np.arange(self.n_episodes):

            # reset environment
            state = self.env.reset()

            # reset trajectory
            self.reset_trajectory()

            # terminal state flag
            complete = False

            # sample episode
            for k in np.arange(n_steps_lim):

                # interrupt if we are in a terminal state
                if complete:
                    break

                # choose action following epsilon greedy policy
                action = self.get_epsilon_greedy_action(ep, state)

                # step dynamics forward
                new_state, r, complete, _ = self.env.step(action)

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

                state = self.states[k]
                action = self.actions[k]
                idx = tuple(state) + (action,)
                g = self.returns[k]

                self.n_table[idx] += 1
                self.q_table[idx] = self.q_table[idx] \
                                  + (g - self.q_table[idx]) / self.n_table[idx]

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
        self.npz_dict['last_n_table'] = self.n_table
        self.npz_dict['last_q_table'] = self.q_table

    def sarsa(self, n_episodes_lim, n_steps_lim, alpha):

        # preallocate information for all epochs
        self.n_episodes = n_episodes_lim
        self.preallocate_episodes()

        # initialize frequency and q-values table
        self.initialize_frequency_table()
        self.initialize_q_table()

        # set epsilons
        self.set_glie_epsilons()

        # episodes
        for ep in np.arange(self.n_episodes):

            # reset environment and choose action
            state = self.env.reset()
            action = self.get_epsilon_greedy_action(ep, state)

            # reset trajectory
            self.reset_trajectory()

            # terminal state flag
            complete = False

            # sample episode
            for k in np.arange(n_steps_lim):

                # interrupt if we are in a terminal state
                if complete:
                    break

                # step dynamics forward
                new_state, r, complete, _ = self.env.step(action)

                # get new action 
                new_action = self.get_epsilon_greedy_action(ep, new_state)

                idx = tuple(state) + (action,)
                idx_new = tuple(new_state) + (new_action,)

                # update frequency table
                self.n_table[idx] += 1

                # update q-values table
                self.q_table[idx] = self.q_table[idx] \
                                  + alpha * (r + self.gamma * self.q_table[idx_new] - self.q_table[idx])

                # save reward
                self.save_reward(r)

                # update state and action
                state = new_state
                action = new_action

            # compute return
            self.compute_discounted_rewards()
            self.compute_returns()

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
        self.npz_dict['last_n_table'] = self.n_table
        self.npz_dict['last_q_table'] = self.q_table

    def sarsa_lambda(self, n_episodes_lim, n_steps_lim, alpha, lam):

        # preallocate information for all epochs
        self.n_episodes = n_episodes_lim
        self.preallocate_episodes()

        # initialize frequency and q-values table
        self.initialize_frequency_table()
        self.initialize_q_table()
        self.initialize_eligibility_traces()

        # set epsilons
        self.set_glie_epsilons()

        # episodes
        for ep in np.arange(self.n_episodes):

            # reset environment and choose action
            state = self.env.reset()
            action = self.get_epsilon_greedy_action(ep, state)

            # reset trajectory
            self.reset_trajectory()

            # terminal state flag
            complete = False

            # sample episode
            for k in np.arange(n_steps_lim):

                # interrupt if we are in a terminal state
                if complete:
                    break

                # step dynamics forward
                new_state, r, complete, _ = self.env.step(action)

                # get new action 
                new_action = self.get_epsilon_greedy_action(ep, new_state)

                idx = tuple(state) + (action,)
                idx_new = tuple(new_state) + (new_action,)

                # update frequency table
                self.n_table[idx] += 1

                # compute temporal difference error
                td_error = r + self.gamma * self.q_table[idx_new] - self.q_table[idx]

                # update eligibility traces table
                self.e_table[idx] += 1

                # update the whole q-value and eligibility traces tables
                self.q_table = self.q_table + alpha * td_error * self.e_table
                self.e_table = self.e_table * self.gamma * lam

                # save reward
                self.save_reward(r)

                # update state and action
                state = new_state
                action = new_action

            # compute return
            self.compute_discounted_rewards()
            self.compute_returns()

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
        self.npz_dict['last_n_table'] = self.n_table
        self.npz_dict['last_q_table'] = self.q_table

    def compute_frequency_table(self):
        self.frequency_table = np.sum(self.last_n_table, axis=2)

    def compute_policy_table(self):
        self.policy_table = np.argmax(self.last_q_table, axis=2)

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
        fig.plot_one_line(self.episodes, self.epsilons)

    def plot_frequency(self):
        self.compute_frequency_table()
        fig = MyFigure(self.dir_path, 'frequency_table')
        fig.axes[0].imshow(self.frequency_table, origin='lower')
        fig.savefig(fig.file_path)

    def plot_policy(self):

        # plot policy table
        self.compute_policy_table()
        fig = MyFigure(self.dir_path, 'policy_table')
        fig.axes[0].imshow(self.policy_table, origin='lower')
        fig.savefig(fig.file_path)

        # plot policy vector field
        x = np.arange(self.env.observation_space[0].n)
        y = np.arange(self.env.observation_space[1].n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        U = np.empty(X.shape)
        V = np.empty(Y.shape)
        for i in x:
            for j in y:
                U[i, j] = self.env.moves[self.policy_table[i, j]][0]
                V[i, j] = self.env.moves[self.policy_table[i, j]][1]
        fig = MyFigure(self.dir_path, 'policy_vector_field')
        fig.axes[0].quiver(X, Y, U, V)
        fig.savefig(fig.file_path)
