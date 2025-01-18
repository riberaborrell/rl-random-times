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

def get_dir_path(env_id: str, algorithm_name: str, param_str: str = '') -> str:

    # relative directory path
    dir_path = os.path.join(env_id, algorithm_name, param_str)

    # create dir path if not exists
    make_dir_path(os.path.join(get_data_dir(), dir_path))

    return dir_path

def get_model_arch_str(**kwargs):
    string = ''
    if 'n_layers' in kwargs.keys():
        string += 'n-layers{:d}_'.format(kwargs['n_layers'])
    if 'd_hidden_layer' in kwargs.keys():
        string += 'hidden-size{:d}_'.format(kwargs['d_hidden_layer'])
    return string

def get_z_estimation_str(**kwargs):
    if 'on-policy' in kwargs['agent'] or kwargs['agent'] == 'model-based-dpg':
        return 'z-estimated_' if kwargs['estimate_z'] else 'z-neglected_'
    else:
        return ''

def get_lr_and_batch_size_str(**kwargs):
    string = ''
    string += 'scheduled-lr_' if 'scheduled_lr' in kwargs.keys() and kwargs['scheduled_lr'] else ''
    string += 'lr{:.1e}_'.format(kwargs['lr']) if 'lr' in kwargs.keys() else ''
    #string += 'lr-decay{:.5f}_'.format(kwargs['lr_decay']) if 'lr_decay' in kwargs.keys() else ''
    string += 'lr-final{:.1e}_'.format(kwargs['lr_final']) if 'lr_final' in kwargs.keys() and kwargs['lr_final'] is not None else ''
    string += 'K{:d}_'.format(int(kwargs['batch_size'])) if 'batch_size' in kwargs.keys() else ''
    string += 'K-z{:d}_'.format(int(kwargs['batch_size_z'])) if 'batch_size_z' in kwargs.keys() else ''

    if 'mini_batch_size' in kwargs.keys() and kwargs['mini_batch_size'] is not None:
        if kwargs['mini_batch_size_type'] == 'constant':
            string += 'mini-K{:d}_'.format(int(kwargs['mini_batch_size']))
        else: #kwargs['mini_batch_size_type'] == 'adaptive'
            string += 'mini-K-adapt{:d}_'.format(int(kwargs['mini_batch_size']))
    return string

def get_iter_str(**kwargs):
    if 'n_episodes' in kwargs.keys():
        string = 'n-episodes{:.0e}_'.format(kwargs['n_episodes'])
    elif 'n_total_steps' in kwargs.keys():
        string = 'n-total-steps{:.0e}_'.format(kwargs['n_total_steps'])
    elif 'n_grad_iterations' in kwargs.keys():
        string = 'n-grad-iter{:.0e}_'.format(kwargs['n_grad_iterations'])
    else:
        string = ''
    return string

def get_seed_str(**kwargs):
    if 'seed' not in kwargs.keys() or not kwargs['seed']:
        string = 'seedNone'.format(kwargs['seed'])
    else:
        string = 'seed{:1d}'.format(kwargs['seed'])
    return string

def get_reinforce_discrete_simple_dir_path(**kwargs):
    # set parameters string
    param_str = 'n-steps-lim{:.0e}_'.format(kwargs['env'].get_wrapper_attr('_max_episode_steps')) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'optim-{}_'.format(kwargs['optim_type']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(kwargs['env_id'], kwargs['agent'], param_str)

def get_reinforce_cont_simple_dir_path(**kwargs):
    # set parameters string
    param_str = 'n-steps-lim{:.0e}_'.format(kwargs['env'].get_wrapper_attr('_max_episode_steps')) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + 'policy-{}_'.format(kwargs['policy_type']) \
              + 'policy-noise{:.2f}_'.format(kwargs['policy_noise']) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'optim-{}_'.format(kwargs['optim_type']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(kwargs['env_id'], kwargs['agent'], param_str)

def get_reinforce_simple_dir_path(**kwargs):
    if kwargs['is_action_continuous']:
        return get_reinforce_cont_simple_dir_path(**kwargs)
    else:
        return get_reinforce_discrete_simple_dir_path(**kwargs)

def get_reinforce_stoch_discrete_dir_path(**kwargs):
    '''
    '''

    # set parameters string
    param_str = 'n-steps-lim{:.0e}_'.format(kwargs['n_steps_lim']) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + '{}_'.format(kwargs['return_type']) \
              + get_z_estimation_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'optim-{}_'.format(kwargs['optim_type']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(kwargs['env_id'], kwargs['agent'], param_str)

def get_reinforce_stoch_cont_dir_path(**kwargs):
    '''
    '''

    # set parameters string
    param_str = 'n-steps-lim{:.0e}_'.format(kwargs['n_steps_lim']) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + 'policy-{}_'.format(kwargs['policy_type']) \
              + 'policy-noise{:.2f}_'.format(kwargs['policy_noise']) \
              + '{}_'.format(kwargs['return_type']) \
              + get_z_estimation_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'optim-{}_'.format(kwargs['optim_type']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(kwargs['env_id'], kwargs['agent'], param_str)

def get_reinforce_stoch_dir_path(**kwargs):
    if kwargs['is_action_continuous']:
        return get_reinforce_stoch_cont_dir_path(**kwargs)
    else:
        return get_reinforce_stoch_discrete_dir_path(**kwargs)

def get_reinforce_det_dir_path(**kwargs):
    '''
    '''

    # set parameters string
    param_str = 'n-steps-lim{:.0e}_'.format(kwargs['n_steps_lim']) \
              + 'gamma{:.3f}_'.format(kwargs['gamma']) \
              + get_model_arch_str(**kwargs) \
              + '{}_'.format(kwargs['return_type']) \
              + get_z_estimation_str(**kwargs) \
              + get_lr_and_batch_size_str(**kwargs) \
              + 'optim-{}_'.format(kwargs['optim_type']) \
              + get_iter_str(**kwargs) \
              + get_seed_str(**kwargs)

    return get_dir_path(kwargs['env_id'], kwargs['agent'], param_str)

