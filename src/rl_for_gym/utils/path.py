import os
import shutil

import numpy as np
import torch

from rl_for_gym.utils.config import PROJECT_ROOT_DIR, DATA_ROOT_DIR

def get_project_dir():
    ''' returns the absolute path of the repository's directory
    '''
    return PROJECT_ROOT_DIR

def get_data_dir():
    ''' returns the absolute path of the repository's data directory
    '''
    return DATA_ROOT_DIR

def make_dir_path(dir_path: str):
    ''' Create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def empty_dir(dir_path: str):
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

def save_data(data: dict, dir_path: str):
    file_path = os.path.join(get_data_dir(), dir_path, 'agent.npz')
    np.savez(file_path, **data)

def load_data(dir_path: str) -> dict:
    try:
        file_path = os.path.join(get_data_dir(), dir_path, 'agent.npz')
        data = dict(np.load(file_path, allow_pickle=True))
        for file_name in data.keys():
            if data[file_name].ndim == 0:
                data[file_name] = data[file_name].item()
        data['dir_path'] = dir_path
        return True, data
    except FileNotFoundError as e:
        print(e)
        return False, None

def save_model(model, dir_path: str, file_name: str):
    torch.save(
        model.state_dict(),
        os.path.join(get_data_dir(), dir_path, file_name),
    )

def load_model(model, rel_dir_path, file_name):
    model.load_state_dict(torch.load(os.path.join(get_data_dir(), rel_dir_path, file_name)))

def get_dir_path(env, algorithm_name: str, param_str: str = '') -> str:

    # relative directory path
    dir_path = os.path.join(env.spec.name, algorithm_name, param_str)

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), dir_path))

    return dir_path
