import math

import torch
import torch.nn.functional as F
from torch import nn


class EncoderCRF(nn.Module):
    """
    A conditional random field with its features provided by a bidirectional RNN
    (GRU by default). As of right now, the model only accepts a batch size of 1
    to represent model parameter updates as a result of stochastic gradient descent.

    Primarily used for part-of-speech tagging in NLP w/ state-of-the-art results.

    In essence a heavily cleaned up version of the article:
    http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

    "Bidirectional LSTM-CRF Models for Sequence Tagging"
    https://arxiv.org/abs/1508.01991

    :param sentence: (seq. length, 1, word embedding size)
    :param sequence (training only): Ground truth sequence label (seq. length)
    :return: Viterbi path decoding score, and sequence.
    """

    def __init__(self, start_tag_index, stop_tag_index, tag_size, embedding_dim, hidden_dim):
        super(EncoderCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.start_tag_index = start_tag_index
        self.stop_tag_index = stop_tag_index
        self.tag_size = tag_size

        self.encoder = nn.GRU(embedding_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True)

        self.tag_projection = nn.Linear(hidden_dim, self.tag_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size))

        self.hidden = self.init_hidden()

    def to_scalar(self, variable):
        return variable.view(-1).data.tolist()[0]

    def argmax(self, vector, dim=1):
        _, index = torch.max(vector, dim)
        return self.to_scalar(index)

    def state_log_likelihood(self, scores):
        max_score = scores.max()
        max_scores = max_score.unsqueeze(0).expand(*scores.size())
        return max_score + torch.log(torch.sum(torch.exp(scores - max_scores)))

    def init_hidden(self):
        return torch.autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, features):
        energies = torch.Tensor(1, self.tag_size).fill_(-10000.)
        energies[0][self.start_tag_index] = 0.

        energies = torch.autograd.Variable(energies)

        for feature in features:
            best_path = []

            # Forward scores + transition scores + emission scores (based on features)
            next_state_scores = energies.expand(*self.transitions.size()) + self.transitions + feature.unsqueeze(
                0).expand(
                *self.transitions.size())

            for index in range(self.tag_size):
                next_possible_states = next_state_scores[index].unsqueeze(0)
                best_path.append(self.state_log_likelihood(next_possible_states))

            energies = torch.cat(best_path).view(1, -1)

        terminal_energy = energies + self.transitions[self.stop_tag_index]
        return self.state_log_likelihood(terminal_energy)

    def encode(self, sentence):
        self.hidden = self.init_hidden()

        outputs, self.hidden = self.encoder(sentence, self.hidden)
        tag_energies = self.tag_projection(outputs.squeeze())
        return tag_energies

    def _score_sentence(self, features, tags):
        score = torch.autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.start_tag_index]), tags])

        for index, feature in enumerate(features):
            score = score + self.transitions[tags[index + 1], tags[index]] + feature[tags[index + 1]]
        score = score + self.transitions[self.stop_tag_index, tags[-1]]
        return score

    def viterbi_decode(self, features):
        backpointers = []

        energies = torch.Tensor(1, self.tag_size).fill_(-10000.)
        energies[0][self.start_tag_index] = 0

        energies = torch.autograd.Variable(energies)
        for feature in features:
            backtrack = []
            best_path = []

            next_state_scores = energies.expand(*self.transitions.size()) + self.transitions

            for index in range(self.tag_size):
                next_possible_states = next_state_scores[index]
                best_candidate_state = self.argmax(next_possible_states, dim=0)

                backtrack.append(best_candidate_state)
                best_path.append(next_possible_states[best_candidate_state])

            energies = (torch.cat(best_path) + feature).view(1, -1)
            backpointers.append(backtrack)

        # Transition to STOP_TAG.
        terminal_energy = energies + self.transitions[self.stop_tag_index]
        best_candidate_state = self.argmax(terminal_energy)
        path_score = terminal_energy[0][best_candidate_state]

        # Backtrack decoded path.
        best_path = [best_candidate_state]
        for backtrack in reversed(backpointers):
            best_candidate_state = backtrack[best_candidate_state]
            best_path.append(best_candidate_state)

        best_path.reverse()
        best_path = best_path[1:]

        return path_score, best_path

    def loss(self, sentence, tags):
        features = self.encode(sentence)
        forward_score = self._forward_alg(features)
        gold_score = self._score_sentence(features, tags)

        return forward_score - gold_score

    def forward(self, sentence):
        features = self.encode(sentence)

        viterbi_score, best_tag_sequence = self.viterbi_decode(features)
        return viterbi_score, best_tag_sequence


class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_size=50, num_mixtures=20):
        super(MixtureDensityNetwork, self).__init__()

        self.projection = nn.Linear(input_dim, hidden_size)

        self.mean_projection = nn.Linear(hidden_size, num_mixtures)
        self.std_projection = nn.Linear(hidden_size, num_mixtures)
        self.weights_projection = nn.Linear(hidden_size, num_mixtures)

    def forward(self, x):
        """
        A model for non-linear data that works off of mixing multiple Gaussian
        distributions together. Uses linear projections of a given input to generate
        a set of N Gaussian models' mixture components, means and standard deviations.

        :param x: (num. samples, input dim.)
        :return: Mixture components, means, and standard deviations
            in the form (num. samples, num. mixtures)
        """
        x = F.tanh(self.projection(x))

        weights = F.softmax(self.weights_projection(x))
        means = self.mean_projection(x)
        stds = torch.exp(self.std_projection(x))

        return weights, means, stds


class MixtureDensityLoss(nn.Module):
    def __init__(self):
        super(MixtureDensityLoss, self).__init__()

    def forward(self, y, weights, mean, std):
        """
        Presents a maximum a-priori objective for a set of predicted means, mixture components,
        and standard deviations to model a given ground-truth 'y'. Modeled using negative log
        likelihood.

        :param y: Non-linear target.
        :param weights: Predicted mixture components.
        :param mean: Predicted mixture means.
        :param std: Predicted mixture standard deviations.
        :return:
        """
        normalization = 1.0 / ((2.0 * math.pi) ** 0.5)
        gaussian_sample = (y.expand_as(mean) - mean) * torch.reciprocal(std)
        gaussian_sample = normalization * torch.reciprocal(std) * torch.exp(-0.5 * gaussian_sample ** 2)

        return -torch.mean(torch.log(torch.sum(weights * gaussian_sample, dim=1)))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from torch import autograd, optim

    num_samples = 2500

    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, num_samples))).T
    r_data = np.float32(np.random.normal(size=(num_samples, 1)))
    x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', alpha=0.3)
    plt.show()

    x = autograd.Variable(torch.from_numpy(x_data).cuda())
    y = autograd.Variable(torch.from_numpy(y_data).cuda())

    criterion = MixtureDensityLoss()
    model = MixtureDensityNetwork().cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        weights, means, stds = model(x)
        loss = criterion(y, weights, means, stds)

        print(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x_test_data = np.float32(np.random.uniform(-15, 15, (1, num_samples))).T
    x_test = autograd.Variable(torch.from_numpy(x_test_data).cuda())

    weights, means, stds = model(x_test)

    results = torch.rand(x_test.size(0), 10).cuda()
    samples = torch.randn(x_test.size(0), 10).cuda()

    for ensemble_index in range(results.size(1)):
        for sample_index in range(results.size(0)):
            accumulated_probability, most_likely_distribution = 0, 0

            mixture_weights = weights[sample_index]

            # Determine best-candidate Gaussian distribution based off mixture components.
            for distribution_index in range(mixture_weights.size(0)):
                accumulated_probability += mixture_weights[distribution_index].data[0]
                if accumulated_probability >= results[sample_index, ensemble_index]:
                    most_likely_distribution = distribution_index
                    break

            # Get distribution mean & std. deviation.
            distribution_mean = means[sample_index, most_likely_distribution].data[0]
            distribution_std = stds[sample_index, most_likely_distribution].data[0]

            # Project a random IID point to the distribution.
            results[sample_index, ensemble_index] = distribution_mean + samples[
                                                                            sample_index, ensemble_index] * distribution_std

    y_test_data = results.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.plot(x_test_data, y_test_data, 'b.', x_data, y_data, 'r.', alpha=0.3)
    plt.show()
