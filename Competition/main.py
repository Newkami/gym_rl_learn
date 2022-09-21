from model.QMIX.arguments import get_common_args, get_mixer_args
from environment.environment import Env
from runner import Runner
if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)
    args.n_actions = 30
    args.n_agents = 10
    args.state_shape = 10
    args.obs_shape = 4
    args.episode_limit = 200
    env = Env(2, 3, 4, 1, 5, 30, 10)
    env.reset()
    runner = Runner(env, args)
    runner.run(0)
