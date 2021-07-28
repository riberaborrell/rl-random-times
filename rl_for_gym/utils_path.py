import numpy as np

from pathlib import Path
import shutil
import os

SOURCE_PATH = Path(os.path.dirname(__file__))
PROJECT_PATH = SOURCE_PATH.parent
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

def make_dir_path(dir_path):
    ''' Create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def empty_dir(dir_path):
    ''' Remove all files in the directory from the given path
    '''
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Reason: {}'.format((file_path, e)))

def get_env_dir_path(env):
    dir_path = os.path.join(
        DATA_PATH,
        env.spec.id,
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_agent_dir_path(env, agent):
    dir_path = os.path.join(
        DATA_PATH,
        env.spec.id,
        agent,
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_mc_prediction_dir_path(env_dir_path, explorable_starts, constant_alpha,
                               alpha, gamma, n_episodes):

    # set agent name
    if not explorable_starts and not constant_alpha:
        agent_name = 'mc-prediction'
    elif not explorable_starts and constant_alpha:
        agent_name = 'mc-prediction-alpha'
    elif explorable_starts and not constant_alpha:
        agent_name = 'mc-prediction-es'
    else:
        agent_name = 'mc-prediction-alpha-es'

    # set alpha string
    if not constant_alpha:
        alpha_str = ''
    else:
        alpha_str = 'alpha_{:1.2f}'.format(alpha)

    # set dir path
    dir_path = os.path.join(
        env_dir_path,
        agent_name,
        'gamma_{:1.2f}'.format(gamma),
        alpha_str,
        'N_{:.0e}'.format(n_episodes),

    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_td_prediction_dir_path(env_dir_path, explorable_starts,
                               gamma, alpha, n_episodes):

    # set agent name
    if not explorable_starts:
        agent_name = 'td-prediction'
    else:
        agent_name = 'td-prediction-es'

    # set dir path
    dir_path = os.path.join(
        env_dir_path,
        agent_name,
        'gamma_{:1.2f}'.format(gamma),
        'alpha_{:1.2f}'.format(alpha),
        'N_{:.0e}'.format(n_episodes),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_mc_dir_path(env_dir_path, explorable_starts, constant_alpha, alpha, gamma,
                    eps_type, n_episodes):
    # set agent name
    if not explorable_starts and not constant_alpha:
        agent_name = 'mc-learning'
    elif not explorable_starts and constant_alpha:
        agent_name = 'mc-learning-alpha'
    elif explorable_starts and not constant_alpha:
        agent_name = 'mc-learning-es'
    else:
        agent_name = 'mc-learning-alpha-es'

    # set alpha string
    if not constant_alpha:
        alpha_str = ''
    else:
        alpha_str = 'alpha_{:1.2f}'.format(alpha)

    # set dir path
    dir_path = os.path.join(
        env_dir_path,
        agent_name,
        'gamma_{:1.2f}'.format(gamma),
        alpha_str,
        'eps_{}'.format(eps_type),
        'N_{:d}'.format(n_episodes),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_sarsa_lambda_dir_path(agent_dir_path, eps_type, alpha, lam, n_episodes):
    dir_path = os.path.join(
        agent_dir_path,
        'eps_{}'.format(eps_type),
        'alpha_{:0.3f}'.format(alpha),
        'lambda_{:0.1f}'.format(lam),
        'N_{:d}'.format(n_episodes),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path

def get_qlearning_dir_path(agent_dir_path, eps_type, alpha, n_episodes):
    dir_path = os.path.join(
        agent_dir_path,
        'eps_{}'.format(eps_type),
        'alpha_{:0.3f}'.format(alpha),
        'N_{:d}'.format(n_episodes),
    )

    # create dir path if not exists
    make_dir_path(dir_path)

    return dir_path
