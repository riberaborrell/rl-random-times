import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np

from rl_for_gym.wrappers.episodic_reacher import EpisodicReacherEnv
from rl_for_gym.utils.base_parser import get_base_parser
from rl_for_gym.utils.evaluate import eval_random_policy_vect_sync
from rl_for_gym.utils.plots import plot_y_per_episode

def make_env(env_id, max_episode_steps=100):
    def _init():
        env = gym.make(env_id)
        env = EpisodicReacherEnv(env, threshold_dist=0.01)
        env = TimeLimit(env, max_episode_steps=int(1e6))
        return env
    return _init


def main():
    args = get_base_parser().parse_args()

    # create vectorized environment
    env = gym.vector.SyncVectorEnv([make_env("Reacher-v5") for _ in range(args.n_episodes)])

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
