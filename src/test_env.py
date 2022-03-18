import gym

from custom_policies.custom_feature_extractor import CustomFeatureExtractor
import wrappers
import custom_envs.custom_robotics
from wrappers.preference_reward.model_wrapper import ModelWrapper

import matplotlib.pyplot as plt
from IPython import display
from pyvirtualdisplay import Display

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

env = gym.make('CustomFetchReachDense-v0')

obs = env.reset()
env.step(1)
width = 128
height = 128
plt.figure(3)
plt.clf()
img = env.render(mode='rgb_array', width=width, height=height)
img = img[int(0.15*height): int(0.65*height), int(0.2*width):int(0.7*width)]
plt.imshow(img)
plt.axis('off')

display.clear_output(wait=True)
display.display(plt.gcf())
plt.savefig('env.png')