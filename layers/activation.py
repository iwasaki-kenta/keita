import torch
import torch.nn.functional as F
from torch import nn


class GatedActivation(nn.Module):
    def __init__(self, num_channels):
        super(GatedActivation, self).__init__()

        self.kernel_size = 1
        self.weights = nn.Parameter(torch.FloatTensor(num_channels, num_channels, self.kernel_size * 2))

    def forward(self, x):
        """
        Conditional Image Generation with PixelCNN Decoders
        http://arxiv.org/abs/1606.05328

        1D gated activation unit that models the forget gates and
        real gates of an activation unit using convolutions.

        :param x: (batch size, # channels, height)
        :return: tanh(conv(Wr, x)) * sigmoid(conv(Wf, x))
        """

        real_gate_weights, forget_gate_weights = self.weights.split(self.kernel_size, dim=2)

        real_gate = F.tanh(F.conv1d(input=x, weight=real_gate_weights, stride=1))
        forget_gate = F.sigmoid(F.conv1d(input=x, weight=forget_gate_weights, stride=1))
        return real_gate * forget_gate


if __name__ == "__main__":
    x = torch.autograd.Variable(torch.rand(32, 128, 5))
    model = GatedActivation(num_channels=128)
    print(model(x).size())
