import gymnasium as gym
import numpy as np

from rl_random_times.po.ppo_core import PPO
from rl_random_times.po.ppo_parser import add_ppo_arguments
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.path import get_hopper_env_str
from rl_random_times.utils.plots import plot_y_per_x

def make_env(env_id, gamma, kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def main():
    parser = get_base_parser()
    add_ppo_arguments(parser)
    parser.add_argument(
        '--healthy-reward',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--forward-reward',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--ctrl-cost',
        type=float,
        default=1e-3,
    )
    args = parser.parse_args()

    assert 'Hopper' in args.env_id, 'This script only works with the Hopper environment.'

    # environment parameters
    kwargs = {}

    # hopper parameters
    kwargs['healthy_reward'] = args.healthy_reward
    kwargs['forward_reward_weight'] = args.forward_reward
    kwargs['ctrl_cost_weight'] = args.ctrl_cost

    # env setup
    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.gamma, kwargs) for i in range(args.n_envs)]
        #[make_env(args.env_id, args.gamma) for i in range(args.batch_size)]
    )
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # environment name
    env_name = get_hopper_env_str(
        args.env_id, args.healthy_reward, args.forward_reward, args.ctrl_cost,
    )

    # PPO agent
    agent = PPO(
        env,
        env_name=env_name,
        n_envs=args.n_envs,
        n_steps_lim=args.n_steps_lim,
        gamma=args.gamma,
        n_total_steps=args.n_total_steps,
        n_layers=args.n_layers,
        d_hidden_layer=args.d_hidden_layer,
        policy_noise_init=args.policy_noise,
        n_mini_batches=args.n_mini_batches,
        lr=args.lr,
        anneal_lr=args.lr,
        update_epochs=args.update_epochs,
        max_grad_norm=args.max_grad_norm,
        optim_type=args.optim_type,
        norm_adv=args.norm_adv,
        gae_lambda=args.gae_lambda,
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
    plot_y_per_x(iters, data['mean_returns'], title='Returns', run_window=10, legend=True)
    plot_y_per_x(iters, data['mean_lengths'], title='Time steps', run_window=10)


if __name__ == "__main__":
    main()
