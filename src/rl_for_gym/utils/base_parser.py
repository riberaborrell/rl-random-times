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
        default=None,
        help='random seed (default: None)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.,
        help='discount factor (default: 1.)',
    )
    parser.add_argument(
        '--n-steps-lim',
        type=int,
        default=None,
        help='Set number of maximum steps for an episode. Default: None',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=1000,
        help='Set number of episodes. Default: 1000',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--scheduled-lr',
        action='store_true',
        help='the step size / learning rate parameter is custom scheduled.',
    )
    parser.add_argument(
        '--lr-final',
        type=float,
        default=None,
        help='Set final learning rate. Default: None',
    )
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=1.,
        help='Set weight decay for the learning rate. Default: 1.',
    )
    parser.add_argument(
        '--n-grad-iterations',
        type=int,
        default=1000,
        help='Set number of gradient iterations. Default: 1000',
    )
    parser.add_argument(
        '--expectation-type',
        choices=['random-time', 'on-policy'],
        default='random-time',
        help='Set type of expectation. Default: random-time',
    )
    parser.add_argument(
        '--return-type',
        choices=['initial-return', 'n-return'],
        default='initial-return',
        help='Set type of return used. Default: initial-return',
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=2,
        help='Set total number of layers. Default: 2',
    )
    parser.add_argument(
        '--d-hidden',
        type=int,
        default=32,
        help='Set dimension of the hidden layers. Default: 32',
    )
    parser.add_argument(
        '--gaussian-policy-type',
        type=str,
        default='learnt-cov',
        choices=['const-cov', 'scheduled', 'learnt-cov'],
        help='Set if the covariance of the stochastic gaussian policy is constant, scheduled, or learnt. Default: const-cov',
    )
    parser.add_argument(
        '--policy-noise',
        type=float,
        default=1.0,
        help='Set factor of scalar covariance matrix. Default: 1.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Set number of trajectories in each batch. Default: 10',
    )
    parser.add_argument(
        '--batch-size-z',
        type=int,
        default=1000,
        help='Set number of trajectories in each batch to estimate the z-factor. Default: 1000',
    )
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=None,
        help='Set mini batch size for on-policy expectations. Default: None',
    )
    parser.add_argument(
        '--mini-batch-size-type',
        choices=['constant', 'adaptive'],
        default='adaptive',
        help='Set type of mini batch size. Constant or adaptive relative to the \
              memory size. Default: constant',
    )
    parser.add_argument(
        '--estimate-z',
        action='store_true',
        help='Estimate the z normalization factor for the spg or dpg gradients.',
    )
    parser.add_argument(
        '--optim-type',
        choices=['sgd', 'adam'],
        default='adam',
        help='Set optimization routine. Default: adam',
    )
    parser.add_argument(
        '--n-envs',
        type=int,
        default=None,
        help='Set number of Envpool environments. Default: None',
    )
    parser.add_argument(
        '--envpool',
        action='store_true',
        help='Set envpool flag. Default: False',
    )
    parser.add_argument(
        '--log-freq',
        type=int,
        default=100,
        help='interval between training status logs (default: 100)',
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load already run agent. Default: False',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Do plots. Default: False',
    )
    parser.add_argument(
        '--live-plot-freq',
        type=int,
        default=None,
        help='Set frequency to live plots. Default: None',
    )
    parser.add_argument(
        '--backup-freq',
        type=int,
        default=100,
        help='Set frequency of backups. Default: 10',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='render the environment',
    )
    parser.add_argument(
        '--time-sleep',
        type=float,
        default=None,
        help='sleep time between environment steps for slowing rendering',
    )
    return parser
