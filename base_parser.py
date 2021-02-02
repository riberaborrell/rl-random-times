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
        '--render',
        dest='render',
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
    return parser
