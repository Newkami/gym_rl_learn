from model.QMIX.arguments import get_common_args, get_mixer_args
from environment import Env
import torch

if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)
    env = Env(10, 10, 10, 10, 20, 10, 10)
    a = env.reset()
    a = torch.tensor(a[0], dtype=torch.float32)

    print(a.shape)