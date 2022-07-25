import numpy as np

from gym import Env, ObservationWrapper, Wrapper, spaces
from torch.utils.tensorboard import SummaryWriter


class ImageAndRobot(ObservationWrapper):

    def __init__(self, env, width=128, height=128):
        super(ImageAndRobot, self).__init__(env)
        self.width = width
        self.height = height
        low, high = (0, 255)
        pixels_space = spaces.Box(shape=(width, height, 3), low=low, high=high, dtype=np.uint8)
        self.observation_space = spaces.Dict(
            image=pixels_space,
            robot=env.observation_space['robot']
        )

    def observation(self, observation):
        obs = {
            'image': self.env.render(mode='rgb_array', width=self.width, height=self.height),
            'robot': observation['robot']
        }
        return obs

class PixelObservation(ObservationWrapper):

    def __init__(self, env, width=128, height=128, add_robot=False):
        super(ObservationWrapper, self).__init__(env)
        self.width = width
        self.height = height
        self.add_robot = add_robot
        low, high = (0, 255)
        pixels_space = spaces.Box(shape=(height, width, 3), low=low, high=high, dtype=np.uint8)
        if self.add_robot:
            self.observation_space = spaces.Dict(
                image=pixels_space,
                robot=env.observation_space['robot']
            )
        else:
            self.observation_space = spaces.Dict(
                image=pixels_space,
            )

    def observation(self, observation):
        if self.add_robot:
            obs = {
                'image': self.env.render(mode='rgb_array', width=self.width, height=self.height),
                'robot': observation['robot']
            }
        else:
            obs = {
                'image': self.env.render(mode='rgb_array', width=self.width, height=self.height)
            }
        return obs

class RemoveGoal(ObservationWrapper):

    def __init__(self, env):
        super(RemoveGoal, self).__init__(env)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,))
        
        obs = env._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                )
            )
        )

    def observation(self, observation):
        return dict(
            observation=observation['observation'], 
            desired_goal=observation['desired_goal']
        )

class AddImage(ObservationWrapper):

    def __init__(self, env, width=128, height=128):
        super(AddImage, self).__init__(env)
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,))
        self.width = width
        self.height = height
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=env.observation_space['desired_goal'],
                achieved_goal=env.observation_space['achieved_goal'],
                observation=env.observation_space['observation'],
                image=spaces.Box(
                    shape=(height, width, 3), low=0, high=255, dtype=np.uint8
                )
            )
        )

    def observation(self, observation):
        obs = {
            'image': self.env.render(mode='rgb_array', width=self.width, height=self.height),
            'observation': observation['observation'],
            'desired_goal': observation['desired_goal'],
            'achieved_goal': observation['achieved_goal']
        }
        return obs

class CropImage(ObservationWrapper):
    def __init__(self, env):
        super(CropImage, self).__init__(env)
        self.width = env.observation_space['image'].shape[1]
        self.height = env.observation_space['image'].shape[0]
        low, high = (0, 255)
        pixels_space = spaces.Box(shape=(int(0.5 * self.height), int(0.5 * self.width), 3), low=low, high=high, dtype=np.uint8)
        space = {k:env.observation_space[k] for k in env.observation_space.spaces.keys()}
        space['image'] = pixels_space
        self.observation_space = spaces.Dict(space)
    
    def observation(self, observation):
        observation['image'] = observation['image'][int(0.15*self.height): int(0.65*self.height), int(0.2*self.width):int(0.7*self.width)]
        return observation

class RemoveRobot(ObservationWrapper):

    def __init__(self, env):
        super(RemoveRobot, self).__init__(env)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=env.observation_space['desired_goal'],
                achieved_goal=env.observation_space['achieved_goal'],
                observation=env.observation_space['observation'],
            )
        )

    def observation(self, observation):
        obs = {
            'observation': observation['observation'],
            'desired_goal': observation['desired_goal'],
            'achieved_goal': observation['achieved_goal']
        }
        return obs

class ConcatDict(ObservationWrapper):
    def __init__(self, env):
        super(ConcatDict, self).__init__(env)
        low = np.concatenate([env.observation_space['observation'].low, env.observation_space['achieved_goal'].low, env.observation_space['desired_goal'].low])
        high = np.concatenate([env.observation_space['observation'].high, env.observation_space['achieved_goal'].high, env.observation_space['desired_goal'].high])
        self.observation_space = spaces.Box(low=low, high=high)

    def observation(self, observation):
        return np.concatenate([observation['observation'], observation['achieved_goal'], observation['desired_goal']])

class DictToImageBox(ObservationWrapper):
    def __init__(self, env):
        super(DictToImageBox, self).__init__(env)
        low = env.observation_space['image'].low
        high = env.observation_space['image'].high
        self.observation_space = spaces.Box(low=low, high=high)

    def observation(self, observation):
        return observation['image']

class CostWrapper(Wrapper):
    def __init__(self, env: Env, cost_factor = 1, tensorboard_log: str=None) -> None:
        super().__init__(env)
        self.tensorboard_log = tensorboard_log
        if tensorboard_log != None:
            self.writer = SummaryWriter(log_dir=tensorboard_log)
        else:
            self.writer = None
        self.reward_sum = 0
        self.cost_sum = 0
        self.logged = False
        self.cost_factor = cost_factor

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward
        self.cost_sum += info['cost']
        self.logged = True
        return observation, reward - self.cost_factor * info['cost'], done, info

    
    def reset(self, **kwargs):
        if self.writer != None and self.logged:
            self.writer.add_scalar('cost/reward', self.reward_sum)
            self.writer.add_scalar('cost/cost', self.cost_sum)
        self.reward_sum = 0
        self.cost_sum = 0
        self.logged = False
        observation = self.env.reset(**kwargs)
        return observation