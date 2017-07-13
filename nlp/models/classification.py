import torch
from torch import nn
import torch.nn.functional as F


class HierarchialNetwork1D(nn.Module):
    """
    A shallow 1D CNN text classification model.
    Sequences are assumed to be 3D tensors (sequence length, batch size, word dim.)
    """

    def __init__(self, embed_dim, num_classes=4):
        super(HierarchialNetwork1D, self).__init__()
        self.layers = []

        first_block = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
        )
        self.layers.append(first_block)

        for layer_index in range(4):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(64)
            )
            self.layers.append(conv_block)

        self.fc = nn.Linear(len(self.layers) * 64, num_classes)

    def forward(self, x):
        # Transpose to the shape (batch size, word dim, sequence length)
        x = x.transpose(0, 1).transpose(1, 2)

        feature_maps = []

        for layer in self.layers:
            x = layer(x)
            feature_maps.append(F.max_pool1d(x, kernel_size=x.size(2)).squeeze())

        features = torch.cat(feature_maps, dim=1)
        features = F.relu(self.fc(features))

        return features

if __name__ == '__main__':
    x = torch.autograd.Variable(torch.rand(32, 100, 300))
    model = HierarchialNetwork1D(embed_dim=300)

    print(model(x).size())
