import gym

from custom_policies.custom_feature_extractor import CustomFeatureExtractor
import wrappers
import custom_envs.custom_robotics

import matplotlib.pyplot as plt
from IPython import display
from pyvirtualdisplay import Display

env = gym.make('CustomFetchReachDense-v0')
env = wrappers.PixelObservation(env)
obs = env.reset()
env.step(1)

plt.figure(3)
plt.clf()
plt.imshow(env.render(mode='rgb_array'))
plt.axis('off')

display.clear_output(wait=True)
display.display(plt.gcf())
plt.savefig('env.png')