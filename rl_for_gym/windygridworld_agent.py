from agent import Agent

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

    def compute_policy_table(self):
        self.policy_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
        ), dtype=np.int32)

        self.frequency_table = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
        ), dtype=np.int32)

        for i in range(self.env.observation_space[0].n):
            for j in range(self.env.observation_space[1].n):

                # get action index of the highest action-value pair
                self.policy_table[i, j] = np.argmax(self.last_q_table[i, j, :])

                # count how many time a state tuple has been visit
                self.frequency_table[i, j] = np.sum(self.last_n_table[i, j, :])
