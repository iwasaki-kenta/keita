"""
Minimizing Mahalanobis distance between related pairs, and maximizing between negative pairs.

A loss typically used for creating a Euclidian embedding space for a wide variety of supervised learning problems.
The original implementation was by Davis King @ Dlib.

PyTorch Implementation: https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4

Made changes such that accuracy is provided on a forward pass as well.
"""

import torch
import torch.nn.functional as F
from torch import nn

from utils.variable import variable


class MahalanobisMetricLoss(nn.Module):
    def __init__(self):
        super(MahalanobisMetricLoss, self).__init__()

    def forward(self, outputs, targets, margin=0.6, extra_margin=0.04):
        """
        :param outputs: Outputs from a network. (batch size, # features)
        :param targets: Target labels. (batch size, 1)
        :param margin: Minimum distance margin between contrasting sample pairs.
        :param extra_margin: Extra acceptable margin.
        :return: Loss and accuracy. Loss is a variable which may have a backward pass performed.
        """

        loss = variable(torch.zeros(1))
        batch_size = outputs.size(0)

        # Compute Mahalanobis distance matrix.
        magnitude = (outputs ** 2).sum(1).expand(batch_size, batch_size)
        squared_matrix = outputs.mm(torch.t(outputs))
        distances = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()

        # Determine number of positive + negative thresholds.
        neg_mask = targets.expand(batch_size, batch_size)
        neg_mask = (neg_mask - neg_mask.transpose(0, 1)) != 0

        num_pairs = (1 - neg_mask).sum()  # Number of pairs.
        num_pairs = (num_pairs - batch_size) / 2  # Number of pairs apart from diagonals.
        num_pairs = num_pairs.data[0]

        negative_threshold = distances[neg_mask].sort()[0][num_pairs].data[0]

        num_right, num_wrong = 0, 0
        for r in range(batch_size):
            for c in range(batch_size):
                x_label = targets[r].data[0]
                y_label = targets[c].data[0]
                mahalanobis_distance = distances[r, c]
                euclidian_distance = torch.dist(outputs[r], outputs[c])

                if x_label == y_label:
                    # Positive examples should be less than (margin - extra_margin).
                    if mahalanobis_distance.data[0] > margin - extra_margin:
                        loss += mahalanobis_distance - (margin - extra_margin)

                    # Compute accuracy w/ Euclidian distance.
                    if euclidian_distance.data[0] < margin:
                        num_right += 1
                    else:
                        num_wrong += 1
                else:
                    # Negative examples should be greater than (margin + extra_margin).

                    if (mahalanobis_distance.data[0] < margin + extra_margin) and (
                                mahalanobis_distance.data[0] < negative_threshold):
                        loss += (margin + extra_margin) - mahalanobis_distance

                    # Compute accuracy w/ Euclidian distance.
                    if euclidian_distance.data[0] < margin:
                        num_wrong += 1
                    else:
                        num_right += 1

        accuracy = num_right / (num_wrong + num_right)
        return loss / (2 * num_pairs), accuracy
