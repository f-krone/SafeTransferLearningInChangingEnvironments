import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import sys

import custom_envs.custom_robotics
import wrappers
from wrappers.preference_reward.model_wrapper import ModelWrapper
from wrappers.preference_reward.preference_reward import PreferenceReward
from custom_policies.custom_feature_extractor import CustomFeatureExtractor

base_folder = '../output'
project_name = 'push-ensemble-sweep-1'

if __name__ == '__main__':
    def train_wrapper():
        run = wandb.init(sync_tensorboard=True)
        config = wandb.config

        #model_wrapper = ModelWrapper(list(map(lambda i: f'{base_folder}/fetch-reach-ensemble/SAC_ensemble_' + str(i), range(3))), obs_keys=['achieved_goal', 'desired_goal', 'observation'])  
        def make_env():
            env = gym.make('FetchPushDense-v1')
            env = Monitor(env)  # record stats such as returns
            #env = PreferenceReward(env, model_wrapper, 4, config.alpha, use_wandb=True)
            #env = wrappers.ImageAndRobot(env, 128, 128)
            #env = wrappers.CropImage(env)
            #env = wrappers.FrameStack(env, stack_size=4, use_3d_conv=config.temporal_merge == '3d_conv', add_robot=True)
            return env
        env = DummyVecEnv([make_env])

        model = SAC('MultiInputPolicy', env, 
            learning_starts=int(1e4), 
            buffer_size=int(5e5),
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(shape=4) + config.noise_mean, np.zeros(shape=4) + config.noise_sigma),
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                online_sampling=True,
                goal_selection_strategy=config.sampling_strategie,
                n_sampled_goal=config.n_sampled_goal
            ),
            verbose=1, tensorboard_log=f'{base_folder}/sweeps/{project_name}/{run.id}/runs')
        model.learn(
            total_timesteps=int(5e5),
            callback=WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f'{base_folder}/sweeps/{project_name}/{run.id}/models',
                verbose=2,
            ))
    train_wrapper()
