import envpool
import gymnasium as gym
import numpy as np

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import eval_random_policy_envpool_sync
from rl_for_gym.utils.path import load_data, save_data, get_dir_path
from rl_for_gym.utils.plots import plot_y_per_episode


def main():
    args = get_base_parser().parse_args()

    kwargs = {}
    if args.n_steps_lim is not None:
        kwargs['max_episode_steps'] = args.n_steps_lim

    # create gym env
    env = envpool.make_gymnasium(args.env_id, num_envs=args.n_episodes, seed=args.seed, **kwargs)

    # run random policy vectorized
    succ, data = eval_random_policy_envpool_sync(
        env,
        args.env_id,
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
