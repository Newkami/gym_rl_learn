from model.QMIX.arguments import get_common_args, get_mixer_args
from environment.environment import Env
from runner import Runner
import numpy as np
if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)
    args.n_actions = 301
    args.n_agents = 100
    args.state_shape = 10
    args.obs_shape = 5
    args.episode_limit = 15
    env = Env(10, 10, 10, 10, 20, 30, 10)
    env.reset()
    runner = Runner(env, args)
    runner.run(0)
