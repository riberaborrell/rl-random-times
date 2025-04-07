import gymnasium as gym

import numpy as np

from rl_random_times.spg.stochastic_pg_simple import ReinforceStochastic
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.plots import plot_y_per_episode

def main():
    args = get_base_parser().parse_args()

    # create gym env 
    env = gym.make(args.env_id, max_episode_steps=args.n_steps_lim)

    # reinforce stochastic agent
    agent = ReinforceStochastic(
        env,
        gamma=args.gamma,
        policy_type=args.gaussian_policy_type,
        policy_noise=args.policy_noise,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden_layer,
        lr=args.lr,
        n_episodes=args.n_episodes,
        seed=args.seed,
        optim_type=args.optim_type,
    )

    # run reinforce with random time horizon 
    succ, data = agent.run_reinforce(
        log_freq=args.log_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    x = np.arange(args.n_episodes)
    plot_y_per_episode(x, data['returns'], run_window=100, title='Returns', legend=True)
    plot_y_per_episode(x, data['time_steps'], run_window=100, title='Time steps')
    plot_y_per_episode(x, data['losses'], run_window=100, title='Losses')


if __name__ == '__main__':
    main()
