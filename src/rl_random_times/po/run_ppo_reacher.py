import gymnasium as gym
import numpy as np

from rl_random_times.po.ppo_core import PPO
from rl_random_times.po.ppo_parser import add_ppo_arguments
from rl_random_times.wrappers.episodic_reacher import EpisodicReacherEnv
from rl_random_times.utils.base_parser import get_base_parser
from rl_random_times.utils.path import get_reacher_env_str
from rl_random_times.utils.plots import plot_y_per_x

def make_env(env_id, threshold_dist, threshold_vel, reward_ctrl_weight):
    def _init():
        env = gym.make(env_id)
        env = EpisodicReacherEnv(env, threshold_dist, threshold_vel,
                                 reward_ctrl_weight=reward_ctrl_weight)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(1e6))
        return env
    return _init

def main():
    parser = get_base_parser()
    add_ppo_arguments(parser)
    parser.add_argument(
        '--threshold-dist',
        type=float,
        default=0.05,
        help='Threshold distance for Episodic Reacher environment',
    )
    parser.add_argument(
        '--threshold-vel',
        type=float,
        default=1.,
        help='Threshold angular velocity for Episodic Reacher environment',
    )
    parser.add_argument(
        '--reward-ctrl-weight',
        type=float,
        default=0.1,
        help='Reward control weight parameter of the Reacher environment.',
    )
    args = parser.parse_args()

    #TODO: add batch size z
    # batch size
    K = args.batch_size #if args.expectation_type == 'random-time' else args.batch_size_z

    # create vectorized environment
    assert 'Reacher' in args.env_id, 'This script only works with the Reacher environment.'
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.threshold_dist, args.threshold_vel, args.reward_ctrl_weight) for _ in range(K)]
    )

    # environment name
    env_name = get_reacher_env_str(
        args.env_id, args.threshold_dist, args.threshold_vel, args.reward_ctrl_weight,
    )

    # PPO agent
    agent = PPO(
        envs,
        env_name=env_name,
        n_steps_lim=envs.envs[0]._max_episode_steps,
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
