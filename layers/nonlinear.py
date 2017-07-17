import math

import torch
import torch.nn.functional as F
from torch import nn


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

        :param x: (nun samples, input dim.)
        :return:
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
