import gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

if __name__ == '__main__':

    for run_index in range(3):
        config = {
            "policy_type": "MultiInputPolicy",
            "total_timesteps": int(1.5e6),
            "env_name": "FetchPushDense-v1",
            "name": "SAC_ensemble_" + str(run_index),
            "learning_starts": int(1e4),
            "buffer_size": int(1.5e6),
            "gamma": 0.95,
            "learning_rate": 0.0002,
            "noise": "OrnsteinUhlenbeckActionNoise",
            "mean": 0.3,
            "sigma": 0.2
        }
        project_name = "fetch-push-ensemble"
        base_folder = "../output"

        def make_env():
            env = gym.make(config["env_name"])
            env = Monitor(env)  # record stats such as returns
            return env

        env = DummyVecEnv([make_env])

        name = config["name"]
        run = wandb.init(
            project=project_name,
            entity="f-krone",
            name=config["name"],
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

        model = SAC(config["policy_type"], env, 
            learning_starts=config["learning_starts"], 
            buffer_size=config["buffer_size"],
            gamma=config["gamma"],
            learning_rate=config["learning_rate"],
            policy_kwargs=dict(
                net_arch=[256, 256]
            ),
            action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(shape=4) + config["mean"], np.zeros(shape=4) + config["sigma"]),
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                online_sampling=True,
                goal_selection_strategy='future',
                n_sampled_goal=4
            ),
            verbose=1, tensorboard_log=f"{base_folder}/{project_name}/{name}_{run.id}/runs")
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"{base_folder}/{project_name}/{name}_{run.id}/models",
                verbose=2,
            ),
        )
        model.save(f"{base_folder}/{project_name}/{name}_{run.id}")
        run.finish()