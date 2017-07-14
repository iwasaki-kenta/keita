import torch
from torch import nn


class OmniglotEncoder(nn.Module):
    def __init__(self, feature_size=64):
        super(OmniglotEncoder, self).__init__()
        self.layers = []

        first_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=feature_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layers.append(first_block)

        for layer_index in range(3):
            block = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=feature_size, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feature_size),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layers.append(block)

        self.fc = nn.Linear(feature_size, feature_size)

    def forward(self, input):
        """
        Matching Networks for One Shot Learning
        https://arxiv.org/abs/1606.04080

        A network specifically for embedding images from the Omniglot dataset.
        Primarily used with the TCML network.

        :param input: (batch size, # channels, height, width)
        :return: 64-dim embedding for a given image_embedding.
        """
        for layer in self.layers:
            input = layer(input)

        output = input.view(input.size(0), -1)
        output = self.fc(output)
        return output


if __name__ == "__main__":
    x = torch.autograd.Variable(torch.rand(32, 3, 28, 28))
    model = OmniglotEncoder()

    print(model(x).size())
