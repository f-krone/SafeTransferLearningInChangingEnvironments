import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import sys

import custom_envs.custom_robotics
import wrappers
from wrappers.preference_reward.model_wrapper import ModelWrapper
from wrappers.preference_reward.preference_reward import PreferenceReward
from custom_policies.custom_feature_extractor import CustomFeatureExtractor

"""sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'rollout/ep_rew_mean',
        'goal': 'maximize'
    },
    'parameters': {
        'gamma': {
            'min' : 0.8,
            'max': 0.99
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.001
        },
        'noise_mean': {
            'min': -0.5,
            'max': 0.5
        },
        'noise_sigma': {
            'min': 0.0,
            'max': 0.5
        },
        'alpha': {
            'min': 0.0,
            'max': 1.0
        },
        #'stack_size': {
        #    'values': [1, 2, 3, 4, 5]
        #},
        'policy_arch': {
            'values': [[1024, 512, 256], [512, 256, 256], [512, 256, 128], [256, 256], [256, 128]]
        },
        'temporal_merge': {
            'values': ['late_fusion', '3d_conv']
        }
    }
}"""

base_folder = '../output'
project_name = 'ensemble-preference-sweep-3'

if __name__ == '__main__':
    def train_wrapper():
        run = wandb.init(sync_tensorboard=True)
        config = wandb.config

        model_wrapper = ModelWrapper(list(map(lambda i: f'{base_folder}/fetch-reach-ensemble/SAC_ensemble_' + str(i), range(3))), obs_keys=['achieved_goal', 'desired_goal', 'observation'])  
        def make_env():
            env = gym.make('CustomFetchReachDense-v0')
            env = Monitor(env)  # record stats such as returns
            env = PreferenceReward(env, model_wrapper, 4, config.alpha, use_wandb=True)
            env = wrappers.ImageAndRobot(env, 128, 128)
            env = wrappers.CropImage(env)
            env = wrappers.FrameStack(env, stack_size=4, use_3d_conv=config.temporal_merge == '3d_conv', add_robot=True)
            return env
        env = DummyVecEnv([make_env])

        model = SAC('MultiInputPolicy', env, 
            learning_starts=int(1e4), 
            buffer_size=int(1.5e5),
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            policy_kwargs=dict(
                features_extractor_class=CustomFeatureExtractor,
                features_extractor_kwargs=dict(custom_cnn=False),
                net_arch=config.policy_arch
            ),
            action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(shape=4) + config.noise_mean, np.zeros(shape=4) + config.noise_sigma),
            verbose=1, tensorboard_log=f'{base_folder}/sweeps/{project_name}/runs/{run.id}')
        model.learn(
            total_timesteps=int(1.5e5),
            callback=WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f'{base_folder}/sweeps/{project_name}/models/{run.id}',
                verbose=2,
            ))
    train_wrapper()

    """if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        print(f'joining sweep with id {sweep_id}')
        wandb.agent(sweep_id, train_wrapper, project=project_name)
    else:
        print('starting a new sweep')
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, train_wrapper)"""
