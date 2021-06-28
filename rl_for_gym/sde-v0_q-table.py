from agent import QLearningAgent
from base_parser import get_base_parser

import gym
import gym_sde
from gym.spaces.box import Box

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('sde-v0')

    # initialize Agent
    agent = QLearningAgent(env, args.gamma)

    # get dir path
    agent.set_dir_path('q-learning')

    # load already run q-learning agent
    if not agent.load():
        return

    # define state space
    state_space = Box(
        low=agent.env.observation_space.low[0],
        high=agent.env.observation_space.high[0],
        shape=(1,),
        dtype=np.float32,
    )

    # discretize state space
    h_state = 0.1
    state_space_h = np.mgrid[[
        slice(state_space.low[0], state_space.high[0] + h_state, h_state),
    ]]
    state_space_h = np.moveaxis(state_space_h, 0, -1)

    # run agent following the q_value table (just exploitation)
    q_values = agent.q_values[-1]

    for ep in np.arange(args.n_episodes_lim):

        # reset environment
        obs = agent.env.reset()
        state = agent.env.state

        # reset trajectory
        agent.reset_rewards()

        # terminal state flag
        complete = False

        for _ in range(args.n_steps_lim):

            # interrupt if we are in a terminal state
            if complete:
                    break

            #if args.do_render:
            #    agent.env.render()


            # interpolate state in our discretized state space
            idx_state = np.argmin(np.abs(state_space_h[:, 0] - state))

            # take action by just exploiting
            action = np.argmax(q_values[idx_state])
            #breakpoint()

            # step dynamics forward
            obs, r, complete, _ = agent.env.step(action)
            state = agent.env.state

            # save reward
            agent.save_reward(r)

        # print running avg
        if args.do_report and ep % 1 == 0:
            msg = agent.log_episodes(ep)
            print(msg)

    agent.compute_discounted_rewards()
    agent.compute_returns()


if __name__ == '__main__':
    main()
