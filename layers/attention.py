"""
Layers to do with attention for both RNN & convolutional models.
"""

import torch
import torch.nn.functional as F
from torch import nn


class BahdanauAttention(nn.Module):
    """
    Applies inner dot-product to the last hidden states of a RNN.
    * contains additional support for individually being applied
    to bidirectional RNN models.

    "Neural Machine Translation by Jointly Learning to Align and Translate"
    https://arxiv.org/abs/1409.0473

    Mode 0: Single projection projection for left & right hidden states.
    Mode 1: Independent projection projections for left & right hidden states.
    Mode 2: Concatenate the hidden neurons of both the left & right hidden states and
        pass them through a single projection.
    """

    def __init__(self, hidden_size, mode=0):
        super(BahdanauAttention, self).__init__()

        self.mode = mode

        if mode == 0:
            self.projection = nn.Linear(hidden_size, hidden_size)
        elif mode == 1:
            self.left_projection = nn.Linear(hidden_size, hidden_size)
            self.right_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, *hidden_states):
        if len(hidden_states) == 1:
            hidden_state = hidden_states[0]
            return F.softmax(F.tanh(self.projection(hidden_state))) * hidden_state
        elif len(hidden_states) == 2:
            left_hidden_state, right_hidden_state = hidden_states
            if self.mode == 0 or self.mode == 1:
                if self.mode == 0:
                    left_attention_weights = F.softmax(F.tanh(self.projection(left_hidden_state)))
                    right_attention_weights = F.softmax(F.tanh(self.projection(right_hidden_state)))
                elif self.mode == 1:
                    left_attention_weights = F.softmax(F.tanh(self.left_projection(left_hidden_state)))
                    right_attention_weights = F.softmax(F.tanh(self.right_projection(right_hidden_state)))

                return left_attention_weights * left_hidden_state, right_attention_weights * right_hidden_state
            elif self.mode == 2:
                hidden_state = torch.cat([left_hidden_state, right_hidden_state], dim=1)
                attention_weights = F.softmax(F.tanh(self.projection(hidden_state)))

                return attention_weights * left_hidden_state, attention_weights * right_hidden_state


class LuongAttention(nn.Module):
    """
    Applies various alignments to the last decoder state of a seq2seq
    model based on all encoder hidden states.

    "Effective Approaches to Attention-based Neural Machine Translation"
    https://arxiv.org/abs/1508.04025

    As outlined by the paper, the three models available are:
    "dot", "general", and "concat". Inputs are expected to
    be time-major first. (seq. length, batch size, hidden dim.)
    """

    def __init__(self, hidden_size, mode="general"):
        super(LuongAttention, self).__init__()

        self.mode = mode

        if mode == "general" or mode == "concat":
            self.projection = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        if mode == "concat":
            self.reduction = nn.Parameter(torch.FloatTensor(hidden_size * 2, hidden_size))

    def forward(self, last_state, encoder_states):
        sequence_length, batch_size, hidden_dim = encoder_states.size()

        last_state = last_state.unsqueeze(0).expand(sequence_length, batch_size, last_state.size(1))
        if self.mode == "dot":
            energies = last_state.dot(encoder_states)
        elif self.mode == "general":
            expanded_projection = self.projection.expand(sequence_length, self.projection.size(0),
                                                         self.projection.size(1))
            energies = last_state.dot(encoder_states.bmm(expanded_projection))
        elif self.mode == "concat":
            expanded_reduction = self.reduction.expand(sequence_length, self.reduction.size(0), self.reduction.size(1))
            expanded_projection = self.projection.expand(sequence_length, self.projection.size(0),
                                                         self.projection.size(1))
            energies = torch.cat([last_state, encoder_states], dim=2).bmm(expanded_reduction)
            energies = energies.bmm(expanded_projection)
        attention_weights = F.softmax(energies)

        return attention_weights * encoder_states


if __name__ == "__main__":
    """
    Bahdanau et al. attention layer.
    """
    context = torch.autograd.Variable(torch.rand(32, 128))
    model = BahdanauAttention(128)
    print(model(context).size())

    left_context, right_context = torch.autograd.Variable(torch.rand(32, 128)), torch.autograd.Variable(
        torch.rand(32, 128))

    left_context, right_context = model(left_context, right_context)
    print(left_context.size(), right_context.size())

    """
    Luong et al. attention layer.
    """
    context = torch.autograd.Variable(torch.rand(32, 128))
    encoder_states = torch.autograd.Variable(torch.rand(100, 32, 128))

    model = LuongAttention(128, mode='concat')
    print(model(context, encoder_states).size())
