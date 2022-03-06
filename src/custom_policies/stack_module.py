import torch as th
import torch.nn as nn
from custom_policies.utils import conv_output_shape

class StackModule(nn.Module):
    def __init__(self, stack_size, in_channels, out_channels, width, height):
        super(StackModule, self).__init__()

        self.stack_size = stack_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        output_width, output_height = width, height
        for _ in range(4):
            output_height, output_width = conv_output_shape((output_height, output_width), 3, 2, 1)
        self.fusion = nn.Sequential(
            nn.Linear(output_width * output_height * 256 * stack_size, out_features=out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        assert input.shape[1] == self.stack_size, (input.shape)
        features = []
        for frame in th.unbind(input, dim=1):
            frame = th.moveaxis(frame, -1, 1)
            output = self.feature_extractor(frame)
            features.append(output)
        x = th.cat(features, dim=1)
        return self.fusion(x)
