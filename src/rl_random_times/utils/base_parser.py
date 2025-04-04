import argparse
from gymnasium import envs

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--env-id',
        #choices=[spec.id for spec in envs.registry.all()],
        default='CartPole-v1',
        help='environment id',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='random seed',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.,
        help='discount factor',
    )
    parser.add_argument(
        '--n-steps-lim',
        type=int,
        default=None,
        help='Set the number of maximum steps to run in each environment per policy rollout',
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=1000,
        help='Set number of episodes',
    )
    parser.add_argument(
        '--n-total-steps',
        type=int,
        default=10**6,
        help='Set number of total steps for the environment',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2,
        help='Set learning rate',
    )
    parser.add_argument(
        '--scheduled-lr',
        action='store_true',
        help='the step size / learning rate parameter is custom scheduled',
    )
    parser.add_argument(
        '--lr-final',
        type=float,
        default=None,
        help='Set final learning rate',
    )
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=1.,
        help='Set weight decay for the learning rate',
    )
    parser.add_argument(
        '--n-grad-iterations',
        type=int,
        default=1000,
        help='Set number of gradient iterations',
    )
    parser.add_argument(
        '--expectation-type',
        choices=['random-time', 'on-policy'],
        default='random-time',
        help='Set type of expectation in the policy gradients. (Random-time) trajectory prespective \
              vs (on-policy) state-space prespective',
    )
    parser.add_argument(
        '--return-type',
        choices=['initial-return', 'n-return'],
        default='initial-return',
        help='Set type of return used. (initial return) G_0 vs (n-step return) G_n',
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=2,
        help='Set total number of layers for a feed-forward NN',
    )
    parser.add_argument(
        '--d-hidden',
        type=int,
        default=32,
        help='Set dimension of the hidden layers',
    )
    parser.add_argument(
        '--gaussian-policy-type',
        type=str,
        default='learnt-cov',
        choices=['const-cov', 'scheduled', 'learnt-cov'],
        help='Choose if the covariance of the stochastic gaussian policy is constant, scheduled, or learnt',
    )
    parser.add_argument(
        '--policy-noise',
        type=float,
        default=1.0,
        help='Set factor of scalar covariance matrix',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Set number of trajectories in each batch',
    )
    parser.add_argument(
        '--batch-size-z',
        type=int,
        default=10,
        help='Set number of trajectories in each batch to estimate the z-factor',
    )
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=None,
        help='Set mini batch size for on-policy state-space expectations',
    )
    parser.add_argument(
        '--mini-batch-size-type',
        choices=['constant', 'adaptive'],
        default='adaptive',
        help='Set type of mini batch size. Constant or adaptive i.e. relative to the \
              memory size',
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
        help='Set optimization routine',
    )
    parser.add_argument(
        '--env-type',
        choices=['gym', 'envpool', 'custom'],
        default='gym',
        help='Set type of vectorized environment',
    )
    parser.add_argument(
        '--n-envs',
        type=int,
        default=1,
        help='Set number of parallel environments',
    )
    parser.add_argument(
        '--log-freq',
        type=int,
        default=100,
        help='interval between training status logs',
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load already run agent',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Do plots',
    )
    parser.add_argument(
        '--live-plot-freq',
        type=int,
        default=None,
        help='Set frequency to live plots',
    )
    parser.add_argument(
        '--backup-freq',
        type=int,
        default=100,
        help='Set frequency of backups.',
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
