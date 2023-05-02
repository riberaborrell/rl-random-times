from base_parser import get_base_parser

import gymnasium as gym

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    if args.render:
        env = gym.make(args.env_id, render_mode='human')
    else:
        env = gym.make(args.env_id)

    # reset environment
    obs, info = env.reset(seed=args.seed)

    # terminal state flag
    done = False

    # truncated flag
    truncated = False

    for _ in range(args.n_steps_lim):

        # interrupt if we are in a terminal state
        if done or truncated:
            break

        # render environment

        # take a random action
        action = env.action_space.sample()

        # step dynamics forward
        obs, r, done, truncated, info = env.step(action)

    env.close()

if __name__ == '__main__':
    main()
