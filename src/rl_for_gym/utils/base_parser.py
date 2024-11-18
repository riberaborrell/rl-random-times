import argparse
from gymnasium import envs

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--env-id',
        #choices=[spec.id for spec in envs.registry.all()],
        default='CartPole-v1',
        help='environment id (default: "CartPole-v1")',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.,
        help='discount factor (default: 1.)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--constant-lr',
        action='store_true',
        help='the step size / learning rate parameter is constant.',
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=0.5,
        help='Set lambda parameter for the lambda Sarsa algorithm. Default: 0.5',
    )
    parser.add_argument(
        '--eps-init',
        type=float,
        default=0.5,
        help='Set probility of picking an action randomly. Default: 0.5',
    )
    parser.add_argument(
        '--eps-decay',
        type=float,
        default=0.98,
        help='Set decay rate of epsilon. Default: 0.98',
    )
    parser.add_argument(
        '--eps-min',
        type=float,
        default=0.,
        help='Set minimum value for epsilon. Default: 0.0',
    )
    parser.add_argument(
        '--eps-max',
        type=float,
        default=1,
        help='Set maximum value for epsilon. Default: 1',
    )
    parser.add_argument(
        '--n-steps-lim',
        type=int,
        default=10**6,
        help='Set number of maximum steps for an episode. Default: 1000',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=1000,
        help='Set number of episodes. Default: 1000',
    )
    parser.add_argument(
        '--n-avg-episodes',
        type=int,
        default=100,
        help='Set number last episodes to averaged the statistics. Default: 100',
    )
    parser.add_argument(
        '--step-sliced-episodes',
        type=int,
        default=10,
        help='Set slice episodes step. Default: 10',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Set number of trajectories in each batch. Default: 10',
    )
    parser.add_argument(
        '--h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--log-freq',
        type=int,
        default=10,
        help='interval between training status logs (default: 10)',
    )
    parser.add_argument(
        '--not-truncate',
        dest='truncate',
        action='store_false',
        help='Do not truncate the episode. Default: True',
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load already run agent. Default: False',
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='save agent results. Default: False',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='render the environment',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--live-plot',
        action='store_true',
        help='Do live plots. Default: False',
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Write report. Default: False',
    )
    return parser
