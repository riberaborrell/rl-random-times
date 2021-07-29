from agent import Agent
from base_parser import get_base_parser
from policies import Policy
from reinforce import reinforce

import gym
from gym.spaces.box import Box

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Pendulum-v0')

    # state space
    high = np.array([np.pi, env.max_speed], dtype=np.float32)
    state_space = Box(
        low=-high,
        high=high,
        dtype=np.float32,
    )
    state_space_dim = state_space.shape[0]

    # discretize state space
    h_state = np.array([0.05, 0.1], dtype=np.float32)
    state_space = np.mgrid[[
        slice(state_space.low[0], state_space.high[0] + h_state[0], h_state[0]),
        slice(state_space.low[1], state_space.high[1] + h_state[1], h_state[1]),
    ]]
    breakpoint()
    state_space_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)

    # discretize action space
    action_space_dim = env.action_space.shape[0]
    h_action = np.array([0.1], dtype=np.float32)
    mgrid_input = []
    for i in range(action_space_dim):
        mgrid_input.append(
        slice(env.action_space.low[i], env.action_space.high[i] + h_action[i], h_action[i])
    )
    action_space_h = np.moveaxis(np.mgrid[mgrid_input], 0, -1)

    breakpoint()

    # initialize q-values table
    q = np.empty(np.state_space_h.shape[:-1])


    # different trajectories
    for ep in np.arange(args.n_episodes):

        # reset environment
        env.reset()

        # interpolate state in our discretized state space

        # pick an action

        # step dynamics forward

        # update q-value table


if __name__ == '__main__':
    main()
