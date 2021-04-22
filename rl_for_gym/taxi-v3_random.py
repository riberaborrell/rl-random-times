from base_parser import get_base_parser

import gym

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make('Taxi-v3')

    env.reset()
    for _ in range(args.n_steps_lim):
        env.render()

        # take a random action
        env.step(env.action_space.sample())

    env.close()

if __name__ == '__main__':
    main()
