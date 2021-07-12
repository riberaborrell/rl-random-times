import argparse
from gym import envs

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--env-id',
        dest='env_id',
        choices=[spec.id for spec in envs.registry.all()],
        default='CartPole-v0',
        help='environment id (default: "CartPole-v0")',
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=1,
        help='random seed (default: 1)',
    )
    parser.add_argument(
        '--gamma',
        dest='gamma',
        type=float,
        default=0.99,
        help='discount factor (default: 0.99)',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--eps-init',
        dest='eps_init',
        type=float,
        default=0.5,
        help='Set probility of picking an action randomly. Default: 0.5',
    )
    parser.add_argument(
        '--eps-decay',
        dest='eps_decay',
        type=float,
        default=0.98,
        help='Set decay rate of epsilon. Default: 0.98',
    )
    parser.add_argument(
        '--eps-min',
        dest='eps_min',
        type=float,
        default=0.,
        help='Set minimum value for epsilon. Default: 0.0',
    )
    parser.add_argument(
        '--eps-max',
        dest='eps_max',
        type=float,
        default=1,
        help='Set maximum value for epsilon. Default: 1',
    )
    parser.add_argument(
        '--n-steps-lim',
        dest='n_steps_lim',
        type=int,
        default=1000,
        help='Set number of maximum steps for an episode. Default: 1000',
    )
    parser.add_argument(
        '--n-episodes-lim',
        dest='n_episodes_lim',
        type=int,
        default=1000,
        help='Set number of episodes. Default: 1000',
    )
    parser.add_argument(
        '--step-sliced-episodes',
        dest='step_sliced_episodes',
        type=int,
        default=10,
        help='Set slice episodes step. Default: 10',
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=10,
        help='Set number of trajectories in each batch. Default: 10',
    )
    parser.add_argument(
        '--render',
        dest='do_render',
        action='store_true',
        help='render the environment',
    )
    parser.add_argument(
        '--log-interval',
        dest='log_interval',
        type=int,
        default=10,
        help='interval between training status logs (default: 10)',
    )
    parser.add_argument(
        '--load',
        dest='load',
        action='store_true',
        help='Load already run agent. Default: False',
    )
    parser.add_argument(
        '--do-plots',
        dest='do_plots',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--do-report',
        dest='do_report',
        action='store_true',
        help='Write report. Default: False',
    )
    return parser
