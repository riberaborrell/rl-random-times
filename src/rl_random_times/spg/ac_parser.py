
def add_ac_arguments(parser):
    parser.add_argument(
        '--actor-lr',
        type=float,
        default=1e-2,
        help='the learning rate for the policy (actor)',
    )
    parser.add_argument(
        '--critic-lr',
        type=float,
        default=1e-2,
        help='the learning rate for the value function (critic)',
    )
    parser.add_argument(
        '--clip-vf-loss',
        action='store_true',
        default=False,
        help='toggles whether or not to use a clipped loss for the value function',
    )
    parser.add_argument(
        '--clip-coef',
        type=float,
        default=0.2,
        help='the value function loss clipping coefficient',
    )
    parser.add_argument(
        '--vf-coef',
        type=float,
        default=0.5,
        help='coefficient of the value function in the loss',
    )

    return parser

