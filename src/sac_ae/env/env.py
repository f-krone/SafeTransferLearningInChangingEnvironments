import sys
sys.path.append("..")

import gym
import numpy as np
from collections import deque
import custom_robotics
import wrappers
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def make_envs(args, is_eval=False, logger=None):
    """env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=True,
        height=args.env_image_size,
        width=args.env_image_size,
        frame_skip=args.action_repeat
    )"""
    env = gym.make(args.domain_name + '-' + args.task_name)
    max_episode_steps = env._max_episode_steps
    env.seed(args.seed)
    if not is_eval:
        def load_model(file_name):
            def make_ensemble_env():
                    env = gym.make('FetchPushDense-v1')
                    return env
            ensemble_env = DummyVecEnv([make_ensemble_env])
            return SAC.load(file_name, ensemble_env)
        model_wrapper = wrappers.ModelWrapper(list(map(lambda i: load_model( f'../../output/fetch-push-ensemble/SAC_ensemble_' + str(i)), range(3))), obs_keys=['achieved_goal', 'desired_goal', 'observation'])
        env = wrappers.PreferenceReward(env, model_wrapper, 4, 1.0, logger=logger)
    env = PixelObservation(
        env,
        height=2*args.env_image_size,
        width=2*args.env_image_size)
    env = CropImage(env)
    env = FrameStack(env, k=args.frame_stack, max_episode_steps=max_episode_steps)
    return env

class PixelObservation(gym.ObservationWrapper):
    def __init__(self, env, width=128, height=128):
        super(PixelObservation, self).__init__(env)
        self.width = width
        self.height = height
        low, high = (0, 255)
        self.observation_space = gym.spaces.Box(shape=(3, height, width), low=low, high=high, dtype=np.uint8)

    def observation(self, observation):
        obs = self.env.render(mode='rgb_array', width=self.width, height=self.height)
        obs = obs.transpose(2, 0, 1).copy()
        return obs

class CropImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(CropImage, self).__init__(env)
        self.width = env.observation_space.shape[2]
        self.height = env.observation_space.shape[1]
        low, high = (0, 255)
        self.observation_space = gym.spaces.Box(shape=(3, int(0.5 * self.height), int(0.5 * self.width)), low=low, high=high, dtype=np.uint8)

    def observation(self, observation):
        observation = observation[:, int(0.15*self.height): int(0.65*self.height), int(0.2*self.width):int(0.7*self.width)]
        return observation


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, max_episode_steps):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)