import gymnasium as gym
import gym_sde_is
import numpy as np

from rl_random_times.po.ppo_core import PPO
from rl_random_times.po.ppo_parser import add_ppo_arguments
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.plots import plot_y_per_x

def main():
    parser = get_base_parser()
    parser.description = 'Run episodic version of the ppo algorithm for the sde \
                          importance sampling environment with a ol toy example.'
    add_ppo_arguments(parser)
    parser.add_argument(
        '--d',
        type=int,
        default=1,
        help='the dimension of the environment',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        nargs='+',
        default=[1.],
        help='the potential barrier parameter',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
        help='the inverse of the temperature',
    )
    args = parser.parse_args()

    # create gym environment
    env = gym.make(
        'sde-is-doublewell-nd-mgf-v0',
        d=args.d,
        dt=0.01,
        beta=args.beta,
        alpha=args.alpha,
        state_init_dist='delta',
        max_episode_steps=int(1e6),
        is_vectorized=True,
    )

    # PPO agent
    agent = PPO(
        env,
        env_name=env.unwrapped.__str__(),
        n_steps_lim=env._max_episode_steps,
        gamma=args.gamma,
        policy_noise_init=args.policy_noise,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden_layer,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        update_epochs=args.update_epochs,
        n_mini_batches=args.n_mini_batches,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        optim_type=args.optim_type,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        seed=args.seed,
        cuda=args.cuda,
        torch_deterministic=args.torch_deterministic,
    )

    # run
    succ, data = agent.run_ppo(
        log_freq=args.log_freq,
        backup_freq=args.backup_freq,
        load=args.load,
    )
    env.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    iters = np.arange(data['n_iterations'] + 1)
    plot_y_per_x(iters, data['mean_returns'], title='Returns', run_window=1, legend=True)
    plot_y_per_x(iters, data['mean_lengths'], title='Time steps', run_window=1)


if __name__ == "__main__":
    main()
