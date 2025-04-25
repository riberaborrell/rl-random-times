import gymnasium as gym
import numpy as np

from rl_random_times.po.ppo_core import PPO
from rl_random_times.po.ppo_parser import add_ppo_arguments
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.plots import plot_y_per_x

def main():
    parser = get_base_parser()
    add_ppo_arguments(parser)
    args = parser.parse_args()

    # environment parameters
    kwargs = {}

    # time horizon
    assert args.n_steps_lim is not None, 'n_steps_lim must be set.'
    kwargs['max_episode_steps'] = args.n_steps_lim

    #TODO: add batch size z
    # batch size
    K = args.batch_size #if args.expectation_type == 'random-time' else args.batch_size_z

    # env setup
    envs = gym.make_vec(args.env_id, num_envs=K, vectorization_mode="sync", **kwargs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # PPO agent
    agent = PPO(
        envs,
        env_name=args.env_id,
        n_steps_lim=args.n_steps_lim,
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
        estimate_z=args.estimate_z,
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
    envs.close()

    # do plots
    if not args.plot or not succ:
        return

    # plot returns and time steps
    iters = np.arange(data['n_iterations'] + 1)
    plot_y_per_x(iters, data['mean_returns'], title='Returns', run_window=1, legend=True)
    plot_y_per_x(iters, data['mean_lengths'], title='Time steps', run_window=1)


if __name__ == "__main__":
    main()
