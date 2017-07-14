import torch
from torch import nn

from text.models.rnn.encoders import BidirectionalEncoder


class SiameseNet(nn.Module):
    def __init__(self, num_classes=2, embed_dim=300, fc_dim=512, hidden_dim=512, encoder=BidirectionalEncoder):
        super(SiameseNet, self).__init__()

        self.encoder = encoder(embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=1)

        # Multiply by 2 for 2x hidden states in a bidirectional encoder.
        self.encoder_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.encoder_dim * 4, fc_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, source_sentences, target_sentences):
        """
        Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
        https://arxiv.org/abs/1705.02364

        A Siamese text classification network made w/ the goal of creating sentence embeddings.

        :param source_sentences:  A tuple of Variable's representing padded sentence tensor batch
            [seq. length, batch size, embed. size] and sentence lengths.
        :param target_sentences:  A tuple of Variable's representing padded sentence tensor batch
            [seq. length, batch size, embed. size] and sentence lengths.
        :return: Embedding. (batch size, # classes)
        """

        u = self.encoder(source_sentences)
        v = self.encoder(target_sentences)

        features = torch.cat((u, v, torch.abs(u - v), u * v), 1)
        return self.classifier(features)

    def encode(self, sentences):
        return self.encoder(sentences)


class LinearNet(nn.Module):
    def __init__(self, num_classes=2, embed_dim=300, fc_dim=512, hidden_dim=512, encoder=BidirectionalEncoder):
        super(LinearNet, self).__init__()

        self.encoder = encoder(embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=1)

        # Multiply by 2 for 2x hidden states in a bidirectional encoder.
        self.encoder_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.encoder_dim, fc_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, source_sentences):
        """
        A text classification network made w/ the goal of creating sentence embeddings.

        :param source_sentences:  A tuple of Variable's representing padded sentence tensor batch
            [seq. length, batch size, embed. size] and sentence lengths.
        :return: Embedding. (batch size, # classes)
        """
        return self.classifier(self.encode(source_sentences))

    def encode(self, sentences):
        return self.encoder(sentences)
