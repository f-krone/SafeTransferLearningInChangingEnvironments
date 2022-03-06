import numpy as np

from gym import ObservationWrapper, spaces


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

    def __init__(self, env, width=128, height=128):
        super(ObservationWrapper, self).__init__(env)
        self.width = width
        self.height = height
        low, high = (0, 255)
        pixels_space = spaces.Box(shape=(width, height, 3), low=low, high=high, dtype=np.uint8)
        self.observation_space = spaces.Dict(
            image=pixels_space,
        )

    def observation(self, observation):
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
                    shape=(width, height, 3), low=0, high=255, dtype=np.uint8
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