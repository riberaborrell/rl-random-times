import pytest

import numpy as np

def pytest_addoption(parser):
    parser.addoption(
        '--n-episodes',
        dest='n_episodes',
        type=int,
        default=1000,
        help='Set number of episodes. Default: 1000',
    )

@pytest.fixture(scope='session')
def n_episodes(request):
    return request.config.getoption('n_episodes')

