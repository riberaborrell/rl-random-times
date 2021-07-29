import gym
import gym.envs as envs

def main():
    env_ids = [spec.id for spec in envs.registry.all()]
    for env_id in env_ids:
        print(env_id)

if __name__ == '__main__':
    main()
