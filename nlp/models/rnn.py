import torch
import torch.nn.functional as F
from torch import nn


class BidirectionalEncoder(nn.Module):
    def __init__(self, embed_dim=50, hidden_dim=300, num_layers=4, dropout=0.1, rnn=nn.GRU, pooling_mode="max"):
        super(BidirectionalEncoder, self).__init__()
        self.pooling_mode = pooling_mode

        self.encoder = rnn(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True,
                           dropout=dropout)

    def forward(self, x):
        """
        A bidirectional RNN encoder. Has support for global max/average pooling.

        :param x: A tuple of Variable's representing padded sentence tensor batch
            [seq. length, batch size, embed. size] and sentence lengths.
        :return: Global max/average pooled embedding from bidirectional RNN encoder of [batch_size, hidden_size]
        """

        sentences, sentence_lengths = x

        # Sort sentences by descending length.
        sorted_sentence_lengths, sort_indices = torch.sort(sentence_lengths, dim=0, descending=True)
        _, unsort_indices = torch.sort(sort_indices, dim=0)

        sorted_sentence_lengths = sorted_sentence_lengths.data
        sorted_sentences = sentences.index_select(1, sort_indices)

        # Handle padding for RNN's.
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_sentence_lengths.numpy())

        # [seq. length, sentence_batch size, 2 * num. layers * num. hidden]
        encoder_outputs = self.encoder(packed_sentences)[0]
        encoder_outputs = nn.utils.rnn.pad_packed_sequence(encoder_outputs)[0]

        # Unsort outputs.
        encoder_outputs = encoder_outputs.index_select(1, unsort_indices)

        # Apply global max/average pooling 1D.
        encoder_outputs = encoder_outputs.transpose(1, 2)
        if self.pooling_mode == "max":
            encoder_outputs = F.max_pool1d(encoder_outputs, kernel_size=encoder_outputs.size(2))
        elif self.pooling_mode == "avg":
            encoder_outputs = F.avg_pool1d(encoder_outputs, kernel_size=encoder_outputs.size(2))

        encoder_outputs = encoder_outputs.squeeze()

        return encoder_outputs


if __name__ == "__main__":
    from nlp.utils import test_sentences

    sentences, sentence_lengths = test_sentences(num_sentences=37)
    sentences = sentences.unsqueeze(2)
    sentences = sentences.expand(sentences.size(0), sentences.size(1), 50)
    sentences = sentences.transpose(0, 1)

    inputs = (torch.autograd.Variable(sentences), torch.autograd.Variable(sentence_lengths))
    model = BidirectionalEncoder(embed_dim=50)

    print(model(inputs).size())
