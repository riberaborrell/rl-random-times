import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np

from rl_for_gym.wrappers.episodic_reacher import EpisodicReacherEnv
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import eval_random_policy
from rl_for_gym.utils.plots import plot_y_per_episode

def main():
    parser = get_base_parser()
    parser.add_argument(
        '--threshold-dist',
        type=float,
        default=0.05,
        help='Threshold distance for Episodic Reacher environment. Default: 0.05',
    )
    parser.add_argument(
        '--threshold-vel',
        type=float,
        default=1.,
        help='Threshold angular velocity for Episodic Reacher environment. Default: 1.',
    )
    parser.add_argument(
        '--reward-ctrl-weight',
        type=float,
        default=0.1,
        help='Reward control weight parameter of the Reacher environment. Default: 0.1',
    )
    args = parser.parse_args()


    # create gym env 
    env = gym.make('Reacher-v5')
    env = EpisodicReacherEnv(env, threshold_dist=args.threshold_dist,
                             threshold_angular_vel=args.threshold_vel, reward_ctrl_weight=args.reward_ctrl_weight)
    env = TimeLimit(env, max_episode_steps=int(1e6))

    # run random policy
    succ, data = eval_random_policy(
        env,
        n_episodes=args.n_episodes,
        seed=args.seed,
        log_freq=args.log_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    x = np.arange(args.n_episodes)
    plot_y_per_episode(x, data['returns'], title='Returns', run_window=100, legend=True)
    plot_y_per_episode(x, data['time_steps'], title='Time steps',run_window=100)

if __name__ == '__main__':
    main()
