from torch import nn
import torch
import torch.nn.functional as F


class BidirectionalEncoder(nn.Module):
    def __init__(self, embed_dim=50, hidden_dim=300, num_layers=4, dropout=0.1, model=nn.GRU):
        super(BidirectionalEncoder, self).__init__()

        self.encoder = model(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True,
                             dropout=dropout)

    def forward(self, x):
        sentences, sentence_lengths = x

        # Sort sentences by descending length.
        sorted_sentence_lengths, sort_indices = torch.sort(sentence_lengths, dim=2, descending=True)
        _, unsort_indices = torch.sort(sort_indices, dim=0)
        sorted_sentences = sentences[sort_indices]

        # Handle padding for RNN's.
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_sentence_lengths)
        encoder_outputs = self.encoder(packed_sentences)[0]  # [seq. length, sentence_batch size, 2 * num. layers * num. hidden]
        encoder_outputs = nn.utils.rnn.pad_packed_sequence(encoder_outputs)[0]

        # Unsort outputs.
        encoder_outputs = encoder_outputs[unsort_indices]

        # TODO: Pooling.

        return encoder_outputs

