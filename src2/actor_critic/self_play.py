import numpy as np

from src2.environment.PaperSoccer import Soccer
from src2.actor_critic.model import Model


def main():
    env = Soccer()
    model = Model(env.observation_space, env.action_space, 1024)

    state = env.reset()
    state = np.reshape(state, (1, ) + env.observation_space.shape)
    res = model.step(state)
    print(res)


if __name__ == '__main__':
    main()
