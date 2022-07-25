from cv2 import add
from gym import Wrapper
from gym import spaces
import numpy as np
from collections import deque


class FrameStack(Wrapper):
    def __init__(self, env, stack_size=4, stack_type='color_channels_first', add_robot=False, image_key:str=None):
        super(FrameStack, self).__init__(env)
        if stack_type not in ['color_channels_first', 'stack_size_first', 'concat_images']:
            print('Invalid stack type selected, reverting to default')
            stack_type = 'color_channels_first'
        self.stack_type = stack_type
        self.stack_size = stack_size
        self.add_robot = add_robot

        self.frames = deque(maxlen=stack_size)
        if self.stack_type == 'stack_size_first':
            self.key = 'frame_stack_2d'
            low = np.repeat(self.observation_space['image'].low[np.newaxis, :], stack_size, axis=0)
            high = np.repeat(self.observation_space['image'].high[np.newaxis, :], stack_size, axis=0)
        elif self.stack_type == 'color_channels_first':
            self.key = 'frame_stack_3d'
            low = np.repeat(self.observation_space['image'].low[np.newaxis, :], stack_size, axis=0)
            high = np.repeat(self.observation_space['image'].high[np.newaxis, :], stack_size, axis=0)
            low = np.moveaxis(low, -1, 0)
            high = np.moveaxis(high, -1, 0)
        elif self.stack_type == 'concat_images':
            self.key='image'
            low = np.repeat(self.observation_space['image'].low, stack_size, axis=-1)
            high = np.repeat(self.observation_space['image'].high, stack_size, axis=-1)

        if image_key != None:
            self.key = image_key

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
        if self.stack_type == 'stack_size_first':
            obs = np.asarray(self.frames)
        if self.stack_type == 'color_channels_first':
            obs = np.asarray(self.frames)
            obs = np.moveaxis(obs, -1, 0)
        elif self.stack_type == 'concat_images':
            obs = np.concatenate(self.frames, axis=-1)
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
