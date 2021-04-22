from agent import QLearningAgent
from base_parser import get_base_parser

import gym

import numpy as np

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Taxi-v3')

    # initialize Agent
    agent = QLearningAgent(env, args.gamma)

    # get dir path
    agent.set_dir_path()

    # load already run q-learning agent
    if not agent.load():
        return
    q_values = agent.q_values

    # run agent following the q_value table (just exploitation)

    # reset env
    state = agent.env.reset()

    # terminal state flag
    complete = False

    for _ in range(args.n_steps_lim):

        # interrupt if we are in a terminal state
        if complete:
                break

        if args.do_render:
            agent.env.render()

        # take action by just exploiting
        action = np.argmax(q_values[state])

        # step dynamics forward
        state, r, complete, _ = agent.env.step(action)

if __name__ == '__main__':
    main()
