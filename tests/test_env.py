import gym.envs as envs

import pytest

class TestGymEnvironments:
    '''
    '''

    @pytest.mark.skip(reason='most of the registered env do not load')
    def test_env_ids(self):
        env_ids = [spec.id for spec in envs.registry.all()]
        for env_id in env_ids:
            env = gym.make(env_id)
