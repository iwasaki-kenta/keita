"""
Layers to do with convolution.
"""

from torch import nn


class SeparableConvolution2D(nn.Module):
    """
    A depth-wise convolution followed by a point-wise convolution.
    WARNING: Very slow! Unoptimized for PyTorch.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(SeparableConvolution2D, self).__init__()

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
