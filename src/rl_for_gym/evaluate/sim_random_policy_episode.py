
import gymnasium as gym

from rl_for_gym.utils.base_parser import get_base_parser

def main():
    args = get_base_parser().parse_args()

    # create gym env 
    if args.render:
        env = gym.make(args.env_id, render_mode='human')
    else:
        env = gym.make(args.env_id)

    # reset environment
    obs, info = env.reset(seed=args.seed)

    for k in range(args.n_steps_lim):

        # take a random action
        action = env.action_space.sample()

        # step dynamics forward
        obs, r, terminated, truncated, info = env.step(action)
        truncated = False if not args.truncate else truncated
        done = terminated or truncated

        # log
        print(k, obs, r, done, truncated)

        if done:
            break

    env.close()

if __name__ == '__main__':
    main()
