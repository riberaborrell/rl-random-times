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
    for _ in range(args.n_steps_lim):
        env.render()

        # take a random action
        action = env.action_space.sample()

        # step dynamics forward
        state, r, complete, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    main()
