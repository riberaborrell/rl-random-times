import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from rl_for_gym.wrappers.episodic_swimmer import EpisodicSwimmerEnv
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import simulate_random_policy_episode

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--threshold-fwd-dist',
        type=float,
        default=1.,
        help='Set forward distance goal in the x-direction. Default: 1.',
    )
    args = parser.parse_args()

    # env parameters
    kwargs = {}

    # render mode
    if args.render:
        kwargs['render_mode'] = 'human'

    # create gym env 
    env = gym.make('Swimmer-v5', **kwargs)
    env = EpisodicSwimmerEnv(env, threshold_fwd_dist=1.)
    env = TimeLimit(env, max_episode_steps=int(1e6))

    # simulate
    simulate_random_policy_episode(env, args.seed,
                                   log_freq=args.log_freq, time_sleep=args.time_sleep)


if __name__ == '__main__':
    main()
