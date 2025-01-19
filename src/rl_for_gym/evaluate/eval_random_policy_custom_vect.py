import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import eval_random_policy_custom_vect
from rl_for_gym.utils.plots import plot_y_per_episode


def main():
    args = get_base_parser().parse_args()

    # create gym env
    env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim, is_vectorized=True)

    # run random policy vectorized
    succ, data = eval_random_policy_custom_vect(
        env,
        batch_size=args.n_episodes,
        seed=args.seed,
        #truncate=args.truncate,
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
    plot_y_per_episode(x, data['time_steps'], title='Time steps', run_window=100)

if __name__ == '__main__':
    main()
