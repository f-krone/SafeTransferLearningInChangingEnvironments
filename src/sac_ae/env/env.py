import sys
sys.path.append("..")

import gym
import numpy as np
from collections import deque
import custom_robotics
import wrappers
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def make_envs(args, is_eval=False, use_state=False, logger=None):
    env = gym.make(args.env_name)
    max_episode_steps = env._max_episode_steps
    env.seed(args.seed)
    if not is_eval and args.pr_files != None:
        def load_model(file_name):
            if args.pr_env != None:
                def make_ensemble_env():
                        env = gym.make(args.pr_env)
                        return env
                ensemble_env = DummyVecEnv([make_ensemble_env])
                return SAC.load(file_name, ensemble_env)
            return SAC.load(file_name)
        model_wrapper = wrappers.ModelWrapper(list(map(lambda i: load_model(args.pr_files + str(i)), range(args.pr_size))), obs_keys=['achieved_goal', 'desired_goal', 'observation'])
        if args.pr_adapt_alpha == 'constant':
            alpha = args.pr_alpha
        else:
            alpha = 'auto'
        env = wrappers.PreferenceReward(env, model_wrapper, max_mse=4, alpha=alpha, internal_reward_as_cost=args.pr_as_cost, logger=logger)
    if not use_state:
        crop_img = args.env_name.__contains__('Fetch') and not args.env_name.__contains__('Bird')
        img_size = 2*args.env_image_size if crop_img else args.env_image_size
        if args.cnn_3dconv:
            env = wrappers.PixelObservation(env, width=img_size, height=img_size, add_robot=args.robot_shape > 0)
            if crop_img:
                env = wrappers.CropImage(env)
            env = wrappers.FrameStack(env, stack_size=args.frame_stack, stack_type='color_channels_first', add_robot=args.robot_shape > 0, image_key='image')
            if args.robot_shape == 0:
                env = wrappers.DictToImageBox(env)
            env._max_episode_steps = max_episode_steps
        else:
            env = PixelObservation(
                env,
                height=img_size,
                width=img_size,
                robot=args.robot_shape > 0)
            if crop_img:
                env = CropImage(env, robot=args.robot_shape > 0)
            env = FrameStack(env, k=args.frame_stack, max_episode_steps=max_episode_steps, robot=args.robot_shape > 0)
    else:
        if args.env_name.__contains__('Custom'):
            env = wrappers.RemoveRobot(env)
        if args.env_name.__contains__('Fetch'):
            env = wrappers.ConcatDict(env)
    if args.cost == 'reward':
        env = wrappers.CostWrapper(env, logger=logger, logger_key_prefix='eval/' if is_eval else 'train/')
    env._max_episode_steps = max_episode_steps
    return env

class PixelObservation(gym.ObservationWrapper):
    def __init__(self, env, width=128, height=128, robot=False):
        super(PixelObservation, self).__init__(env)
        self.robot = robot
        self.width = width
        self.height = height
        low, high = (0, 255)
        pixel_space = gym.spaces.Box(shape=(3, height, width), low=low, high=high, dtype=np.uint8)
        if robot:
            self.observation_space = gym.spaces.Dict(
                image=pixel_space,
                robot=env.observation_space['robot']
            )
        else:
            self.observation_space = pixel_space

    def observation(self, observation):
        obs = self.env.render(mode='rgb_array', width=self.width, height=self.height)
        obs = obs.transpose(2, 0, 1).copy()
        if self.robot:
            return {
                'image': obs,
                'robot': observation['robot']
            }
        else:
            return obs

class CropImage(gym.ObservationWrapper):
    def __init__(self, env, robot=False):
        super(CropImage, self).__init__(env)
        self.robot = robot
        self.width = env.observation_space['image'].shape[2] if robot else env.observation_space.shape[2]
        self.height = env.observation_space['image'].shape[1] if robot else env.observation_space.shape[1]
        low, high = (0, 255)
        pixel_space = gym.spaces.Box(shape=(3, int(0.5 * self.height), int(0.5 * self.width)), low=low, high=high, dtype=np.uint8)
        if robot:
            self.observation_space = gym.spaces.Dict(
                image=pixel_space,
                robot=env.observation_space['robot']
            )
        else:
            self.observation_space = pixel_space

    def observation(self, observation):
        obs = observation['image'] if self.robot else observation
        obs = obs[:, int(0.15*self.height): int(0.65*self.height), int(0.2*self.width):int(0.7*self.width)]
        if self.robot:
            return {
                'image': obs,
                'robot': observation['robot']
            }
        else:
            return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, max_episode_steps, robot=False):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self.robot = robot
        shp = env.observation_space['image'].shape if robot else env.observation_space.shape
        pixel_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space['image'].dtype if robot else env.observation_space.dtype
        )
        if robot:
            self.observation_space = gym.spaces.Dict(
                image=pixel_space,
                robot=env.observation_space['robot']
            )
        else:
            self.observation_space = pixel_space
        self._max_episode_steps = max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs['image'] if self.robot else obs)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs['image'] if self.robot else obs)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        assert len(self._frames) == self._k
        pixel =  np.concatenate(list(self._frames), axis=0)
        if self.robot:
            return {
                'image': pixel,
                'robot': obs['robot']
            }
        else:
            return pixel