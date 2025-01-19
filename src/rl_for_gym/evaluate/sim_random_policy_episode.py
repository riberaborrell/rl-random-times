
import gymnasium as gym

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import simulate_random_policy_episode

def main():
    args = get_base_parser().parse_args()

    # env parameters
    kwargs = {}

    # render mode
    if args.render:
        kwargs['render_mode'] = 'human'

    # create gym env 
    env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, **kwargs)

    # simulate
    simulate_random_policy_episode(env, args.seed,
                                   log_freq=args.log_freq, time_sleep=args.time_sleep)


if __name__ == '__main__':
    main()
