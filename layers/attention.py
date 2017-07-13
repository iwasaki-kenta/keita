"""
Layers to do with attention for both RNN & convolutional models.
"""

import torch
import torch.nn.functional as F
from torch import nn


class BidirectionalAttention(nn.Module):
    """
    Applies inner dot-attention to hidden states of a bidirectional RNN.

    Mode 0: Single attention projection for left & right hidden states.
    Mode 1: Independent attention projections for left & right hidden states.
    Mode 2: Concatenate the hidden neurons of both the left & right hidden states and
        pass them through a single projection.
    """

    def __init__(self, hidden_size, mode=0):
        super(BidirectionalAttention, self).__init__()

        self.mode = mode

        if mode == 0:
            self.attention = nn.Linear(hidden_size, hidden_size)
        elif mode == 1:
            self.left_attention = nn.Linear(hidden_size, hidden_size)
            self.right_attention = nn.Linear(hidden_size, hidden_size)

    def forward(self, left_hidden_state, right_hidden_state):
        if self.mode == 0 or self.mode == 1:
            if self.mode == 0:
                left_attention_weights = F.softmax(F.tanh(self.attention(left_hidden_state)))
                right_attention_weights = F.softmax(F.tanh(self.attention(right_hidden_state)))
            elif self.mode == 1:
                left_attention_weights = F.softmax(F.tanh(self.left_attention(left_hidden_state)))
                right_attention_weights = F.softmax(F.tanh(self.right_attention(right_hidden_state)))

            return left_attention_weights * left_hidden_state, right_attention_weights * right_hidden_state
        elif self.mode == 2:
            hidden_state = torch.cat([left_hidden_state, right_hidden_state], dim=1)
            attention_weights = F.softmax(F.tanh(self.attention(hidden_state)))

            return attention_weights * left_hidden_state, attention_weights * right_hidden_state
