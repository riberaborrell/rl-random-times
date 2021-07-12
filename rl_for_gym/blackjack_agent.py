from agent import Agent

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

    def mc_learning(self, n_episodes_lim, n_steps_lim, lr, do_render=False):

        # preallocate information for all epochs
        self.n_episodes = n_episodes_lim
        self.preallocate_episodes()

        # set epsilons
        self.set_glie_epsilons()

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

            print(self.epsilons[ep])

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

                # render observations
                if do_render:
                    self.env.render()

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

                state = self.states[k]
                action = self.actions[k]
                state_action_idx = tuple(state) + (action,)
                g = self.returns[k]

                n_table[state_action_idx] += 1
                q_table[state_action_idx] = q_table[state_action_idx] \
                                          + (g - q_table[state_action_idx]) / n_table[state_action_idx]


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
        self.npz_dict['last_q_table'] = q_table
        self.npz_dict['last_n_table'] = n_table

    def compute_stick_hit_tables(self):
        self.stick_hit_table_ua = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
        ), dtype=bool)

        self.stick_hit_table_nua = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
        ), dtype=bool)

        self.frequency_ua = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
        ), dtype=np.int32)

        self.frequency_nua = np.zeros((
            self.env.observation_space[0].n,
            self.env.observation_space[1].n,
        ), dtype=np.int32)

        for i in range(self.env.observation_space[0].n):
            for j in range(self.env.observation_space[1].n):

                # get action index of the highest action-value pair
                self.stick_hit_table_nua[i, j] = np.argmax(self.last_q_table[i, j, 0, :])
                self.stick_hit_table_ua[i, j] = np.argmax(self.last_q_table[i, j, 1, :])

                # count how many time a state tuple has been visit
                self.frequency_nua[i, j] = np.sum(self.last_n_table[i, j, 0, :])
                self.frequency_ua[i, j] = np.sum(self.last_n_table[i, j, 1, :])
