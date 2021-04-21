from agent import Agent
from base_parser import get_base_parser
from policies import Policy
from reinforce import reinforce

import gym

def get_parser():
    parser = get_base_parser()
    parser.description = ''
    return parser

def main():
    args = get_parser().parse_args()

    # create gym env 
    env = gym.make(args.env_id)

    # create agent
    agent = Agent(env, args.gamma)

    # get policy
    pe = Policy(env)

    # start reinforce
    reinforce(env, agent, pe, args.lr, num_episodes=args.n_episodes,
              batch_size=args.batch_size, do_render=args.do_render)

if __name__ == '__main__':
    main()
