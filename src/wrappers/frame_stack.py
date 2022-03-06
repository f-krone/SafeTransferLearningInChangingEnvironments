from cv2 import add
from gym import Wrapper
from gym import spaces
import numpy as np
from collections import deque


class FrameStack(Wrapper):
    def __init__(self, env, stack_size=4, use_3d_conv=True, add_robot=False):
        super(FrameStack, self).__init__(env)
        self.stack_size = stack_size
        self.use_3d_conv = use_3d_conv
        self.add_robot = add_robot
        self.key = 'frame_stack_3d' if use_3d_conv else 'frame_stack_2d'

        self.frames = deque(maxlen=stack_size)

        low = np.repeat(self.observation_space['image'].low[np.newaxis, ...], stack_size, axis=0)
        high = np.repeat(self.observation_space['image'].high[np.newaxis, ...], stack_size, axis=0)

        if use_3d_conv:
            low = np.moveaxis(low, -1, 0)
            high = np.moveaxis(high, -1, 0)

        pixels_space = spaces.Box(low=low, high=high, dtype=self.observation_space['image'].dtype)

        if add_robot:
            self.observation_space = spaces.Dict({
                    self.key: pixels_space,
                    'robot':env.observation_space['robot']
                })
        else:
            self.observation_space = spaces.Dict({
                    self.key: pixels_space,
                })

    def observation(self, observation):
        assert len(self.frames) == self.stack_size, (len(self.frames), self.stack_size)
        obs = np.asarray(self.frames)
        if self.use_3d_conv :
            obs = np.moveaxis(obs, -1, 0)
        if self.add_robot:
            return dict({self.key: obs, 'robot': observation['robot']})
        return dict({self.key: obs})

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation['image'])
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation['image']) for _ in range(self.stack_size)]
        return self.observation(observation)
