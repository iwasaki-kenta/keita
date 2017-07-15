"""
Layers to do with convolution.
"""

from torch import nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """
    A depth-wise convolution followed by a point-wise convolution.
    WARNING: Very slow! Unoptimized for PyTorch.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.batch_norm_in = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)

        self.batch_norm_out = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.batch_norm_in(x)
        x = self.activation(x)

        x = self.pointwise(x)
        x = self.batch_norm_out(x)
        x = self.activation(x)
        return x


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inputs):
        """
        A 1D dilated convolution w/ padding such that the output
        is the same size as the input.

        :param inputs: (batch size, # channels, height)
        :return: (batch size, # channels, height)
        """
        x = F.pad(inputs.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

        self.left_padding = dilation * ((kernel_size[0] if type(kernel_size) == tuple else kernel_size) - 1)

    def forward(self, inputs):
        """
        A 2D dilated convolution w/ padding such that the output
        is the same size as the input.

        Remember that in PyTorch, all sizes are height-axis-major.

        :param inputs: (batch size, # channels, height, width)
        :return: (batch size, # channels, height, width)
        """
        x = F.pad(inputs, (self.left_padding, 0, 0, 0))

        return super(CausalConv2d, self).forward(x)


if __name__ == "__main__":
    import torch

    image = torch.arange(1, 21).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    image = torch.autograd.Variable(image)

    layer = CausalConv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), dilation=4)
    layer.weight.data.fill_(1)

    print(image.data.numpy())
    print(layer(image).round().data.numpy())