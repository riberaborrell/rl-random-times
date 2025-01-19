import gymnasium as gym
import numpy as np

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import eval_random_policy_vect_sync
from rl_for_gym.utils.plots import plot_y_per_episode


def main():
    args = get_base_parser().parse_args()

    # create gym env
    env = gym.make_vec(args.env_id, num_envs=args.n_episodes, vectorization_mode="sync")

    # set max episode steps
    if args.n_steps_lim is not None:
        for i in range(env.num_envs):
            env.envs[i]._max_episode_steps = args.n_steps_lim

    # run random policy vectorized
    succ, data = eval_random_policy_vect_sync(
        env,
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
