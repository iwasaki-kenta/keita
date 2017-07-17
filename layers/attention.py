"""
Layers to do with attention for both RNN & convolutional models.
"""

import torch
import torch.nn.functional as F
from torch import nn


class BahdanauAttention(nn.Module):
    """
    Applies inner dot-product to the last hidden state of a RNN.
    * contains additional modes for being applied in various ways
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

    As outlined by the paper, the three alignment methods available are:
    "dot", "general", and "concat". Inputs are expected to be time-major
    first. (seq. length, batch size, hidden dim.)

    Both encoder and decoders are expected to have the same hidden dim.
    as well, which is not specifically covered by the paper.
    """

    def __init__(self, hidden_size, mode="general"):
        super(LuongAttention, self).__init__()

        self.mode = mode

        if mode == "general":
            self.projection = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        elif mode == "concat":
            self.reduction = nn.Parameter(torch.FloatTensor(hidden_size * 2, hidden_size))
            self.projection = nn.Parameter(torch.FloatTensor(hidden_size, 1))

    def forward(self, last_state, states):
        sequence_length, batch_size, hidden_dim = states.size()

        last_state = last_state.unsqueeze(0).expand(sequence_length, batch_size, last_state.size(1))
        if self.mode == "dot":
            energies = last_state * states
            energies = energies.sum(dim=2).squeeze()
        elif self.mode == "general":
            expanded_projection = self.projection.expand(sequence_length, self.projection.size(0),
                                                         self.projection.size(1))
            energies = last_state * states.bmm(expanded_projection)
            energies = energies.sum(dim=2).squeeze()
        elif self.mode == "concat":
            expanded_reduction = self.reduction.expand(sequence_length, self.reduction.size(0), self.reduction.size(1))
            expanded_projection = self.projection.expand(sequence_length, self.projection.size(0),
                                                         self.projection.size(1))
            energies = F.tanh(torch.cat([last_state, states], dim=2).bmm(expanded_reduction))
            energies = energies.bmm(expanded_projection).squeeze()
        attention_weights = F.softmax(energies)

        return attention_weights


class BilinearAttention(nn.Module):
    """
    Creates a bilinear transformation between a decoder hidden state
    and a sequence of encoder/decoder hidden states. Specifically used
    as a form of inter-attention for abstractive text summarization.

    "A Deep Reinforced Model for Abstractive Summarization"
    https://arxiv.org/abs/1705.04304
    https://einstein.ai/research/your-tldr-by-an-ai-a-deep-reinforced-model-for-abstractive-summarization

    Hidden state sequences alongside a given target hidden state
    are expected to be time-major first. (seq. length, batch size, hidden dim.)

    Encoder and decoder hidden states may have different hidden dimensions.
    """

    def __init__(self, hidden_size, encoder_dim=None):
        super(BilinearAttention, self).__init__()

        self.encoder_dim = hidden_size if encoder_dim is None else encoder_dim

        self.projection = nn.Parameter(
            torch.FloatTensor(hidden_size, self.encoder_dim))

    def forward(self, last_state, states):
        if len(states.size()) == 2: states = states.unsqueeze(0)

        sequence_length, batch_size, state_dim = states.size()

        transformed_last_state = last_state @ self.projection
        transformed_last_state = transformed_last_state.expand(sequence_length, batch_size, self.encoder_dim)
        transformed_last_state = transformed_last_state.transpose(0, 1).contiguous()
        transformed_last_state = transformed_last_state.view(batch_size, -1)

        states = states.transpose(0, 1).contiguous()
        states = states.view(batch_size, -1)

        energies = transformed_last_state * states
        energies = energies.sum(dim=1)

        attention_weights = F.softmax(energies)

        return attention_weights


if __name__ == "__main__":
    """
    Bahdanau et al. attention layer.
    """
    context = torch.autograd.Variable(torch.rand(32, 128))
    model = BahdanauAttention(hidden_size=128)
    print("Bahdanau et al. single hidden state size:", model(context).size())

    left_context, right_context = torch.autograd.Variable(torch.rand(32, 128)), torch.autograd.Variable(
        torch.rand(32, 128))

    left_context, right_context = model(left_context, right_context)
    print("Bahdanau et al. bidirectional hidden state size:", left_context.size(), right_context.size())

    """
    Luong et al. attention layer.
    """
    context = torch.autograd.Variable(torch.rand(32, 128))
    encoder_states = torch.autograd.Variable(torch.rand(100, 32, 99))

    for mode in ["dot", "general", "concat"]:
        model = LuongAttention(hidden_size=128, mode=mode)
        print("Luong et al. mode %s attended sequence state size: %s" % (
            mode, str(model(context, encoder_states).size())))

    """
    Paulus et al. attention layer.
    """
    decoder_state = torch.autograd.Variable(torch.rand(32, 128))
    decoder_states = torch.autograd.Variable(torch.rand(3, 32, 128))

    decoder_attention = BilinearAttention(hidden_size=128)
    decoder_attention_weights = decoder_attention(decoder_state, decoder_states)
    print("Paulus et al. attended decoder size:", decoder_attention_weights.size())

    encoder_states = torch.autograd.Variable(torch.rand(100, 32, 99))

    encoder_attention = BilinearAttention(hidden_size=128, encoder_dim=99)
    encoder_attention_weights = encoder_attention(decoder_state, encoder_states)
    print("Paulus et al. attended encoder size:", encoder_attention_weights.size())

    encoder_attention_weights = encoder_attention_weights.expand(*decoder_state.size())
    decoder_attention_weights = decoder_attention_weights.expand(*decoder_state.size())

    final_context_vector = torch.cat(
        [decoder_state, decoder_attention_weights * decoder_state, encoder_attention_weights * decoder_state])
    print("Paulus et al. final context vector size:", final_context_vector.size())