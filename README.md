# Keita: A PyTorch Toolkit

## Description

A couple of utilities, dataset loaders, and layers compiled and created by me which I believe everyone should have access to.

Noticed that I've been writing way too many boilerplate code for my work and personal projects, so feel free to check out or utilize any of my code here.

I cannot guarantee fixing any bugs whatsoever; though if you have any you'd like to report then feel free to file an issue/pull request. Feedback definitely is appreciated though!

In terms of code organization, I would urge that I myself am not a fan of using huge repositories of highly un-maintained, dependant code and thus intend to keep this repository as modular as possible (for incorporation of some of my modules into your program).

I intend to make the code as clean as possible, and keep the code style consistent and developer-friendly.

## Dependencies

PyTorch, TorchVision, and the bleeding edge build version of TorchText are needed to use this library.

## Contents

- General deep metric learning losses. (mahalonobis-distance hard negative mining)
- Extended convolution layer support. (separable convolutions)
- Convolution/recurrent-based inter-attention layers (additive, dot-product)
- Convolution/recurrent-based text classification models.
- Convolution/recurrent-based sentence embedding models.
- TorchText extensions for training (test/validation dataset split, word embeddings)