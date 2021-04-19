from agent import Agent
from base_parser import get_base_parser

import gym

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Pendulum-v0')

    env.reset()
    for _ in range(args.n_steps):
        env.render()

        # take a random action
        env.step(env.action_space.sample())

        print(env.state)

    env.close()

if __name__ == '__main__':
    main()
