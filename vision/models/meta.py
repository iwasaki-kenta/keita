import torch
import torch.nn.functional as F
from torch import nn

from layers.activation import GatedActivation
from layers.convolution import CausalConv1d


class TemporalDenseBlock(nn.Module):
    def __init__(self, in_channels, hidden_size=128, dilation=1):
        super(TemporalDenseBlock, self).__init__()

        self.conv1 = CausalConv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=2, dilation=dilation)
        self.conv2 = CausalConv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=2, dilation=dilation)
        self.conv3 = CausalConv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=2, dilation=dilation)

        self.gate1 = GatedActivation(hidden_size)
        self.gate2 = GatedActivation(hidden_size)
        self.gate3 = GatedActivation(hidden_size)

    def forward(self, x):
        """
        A 1D dilated causal convolution dense block for TCML.
        Contains residual connections for the 2nd and 3rd convolution.

        :param x: (batch size, # channels, seq. length)
        :return: (batch size, # channels + 128, seq. length)
        """
        features = self.gate1(self.conv1(x))
        features = self.gate2(self.conv2(features) + features)
        features = self.gate3(self.conv3(features) + features)

        outputs = torch.cat([features, x], dim=1)
        return outputs


class TCML(nn.Module):
    def __init__(self, feature_dim, num_classes=3):
        super(TCML, self).__init__()

        self.dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        self.dense_blocks = [TemporalDenseBlock(feature_dim + 128 * index, hidden_size=128, dilation=dilation) for
                             index, dilation in
                             enumerate(self.dilations)]

        self.conv1 = nn.Conv1d(in_channels=feature_dim + 128 * len(self.dilations), out_channels=512, kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, inputs):
        """
        Meta-Learning with Temporal Convolutions
        https://arxiv.org/abs/1707.03141

        :param inputs: (batch size, # channels, height)
        :return: (batch size, num. classes, height)
        """

        for index, block in enumerate(self.dense_blocks):
            features = block(inputs if index == 0 else features)

        features = F.relu(self.conv1(features), inplace=True)
        features = self.conv2(features)

        return features


if __name__ == "__main__":
    image_embedding = torch.autograd.Variable(torch.rand(32, 64, 5))
    model = TCML(feature_dim=image_embedding.size(1), num_classes=3)
    print(model(image_embedding).size())
