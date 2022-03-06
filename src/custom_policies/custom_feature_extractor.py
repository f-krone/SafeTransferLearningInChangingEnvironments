import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from .stack_module import StackModule
from custom_policies.utils import conv_output_shape_3d

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, custom_cnn: bool = True):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                if custom_cnn:
                    extractors[key] = nn.Sequential(
                        nn.Conv2d(subspace.shape[0], 16, kernel_size=3, padding = 1),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, padding = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, padding = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, padding = 1),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=(subspace.shape[1] // 4, subspace.shape[2] // 4)),
                        nn.Flatten()
                    )
                    total_concat_size += 128
                else:
                    extractors[key] = NatureCNN(subspace, 256)
                    total_concat_size += 256
            elif key == "frame_stack_3d":
                #for a stack of 4 frames, this produces an output of a single frame
                output_height, output_width, output_depth = subspace.shape[2], subspace.shape[3], subspace.shape[1]
                output_height, output_width, output_depth = conv_output_shape_3d((output_height, output_width, output_depth), 3, 2, 1)
                output_height, output_width, output_depth = conv_output_shape_3d((output_height, output_width, output_depth), 3, 2, 1)
                output_height, output_width, output_depth = conv_output_shape_3d((output_height, output_width, output_depth), 3, (2, 2, 1), 1)
                extractors[key] = nn.Sequential(
                    nn.Conv3d(subspace.shape[0], 16, kernel_size=3, stride = 2, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(16, 32, kernel_size=3, stride = 2, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(32, 64, kernel_size=3, stride = (1, 2, 2), padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(output_height * output_width * output_depth * 64, 1024),
                    nn.ReLU()
                )
                total_concat_size += 1024
            elif key == "frame_stack_2d":
                extractors[key] = StackModule(subspace.shape[0], subspace.shape[3], 1024, subspace.shape[1], subspace.shape[2])
                total_concat_size += 1024
            elif key == "robot":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(get_flattened_obs_dim(subspace), 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU()
                )
                total_concat_size += 128

            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)