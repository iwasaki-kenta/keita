# Keita: A PyTorch Toolkit

## Description

A couple of utilities, dataset loaders, and layers compiled and created by me which I believe everyone should have access to.

Noticed that I've been writing way too much boilerplate code for work alongside my personal projects, and so decided to compact and package it up for open usage.

I cannot guarantee fixing potential bugs you may find whatsoever; though if you'd like to report any then feel free to file an issue/pull request and I'll try my luck on it. Feedback and suggestions are definitely appreciated!

In terms of code organization, I would urge that I myself am not a fan of using huge repositories of highly un-maintained, dependant code and thus intend to keep this repository as modular as possible (for incorporation of some of my modules into your program).

I also intend to make the code as clean as possible, and keep the code style consistent and developer-friendly (clear variable names, simple references to utility functions).

## Dependencies

PyTorch, TorchVision, TQDM, and the bleeding edge build version of TorchText are needed to use this library.

## Contents

- Deep metric learning losses. (mahalonobis-distance hard negative mining)
- Meta-learning models. (temporal convolution meta-learner)
- Activation unit layers. (gated activation unit for PixelCNN)
- Extended convolution layer support. (separable convolutions)
- Convolution/recurrent-based inter-attention layers (additive, dot-product)
- Convolution/recurrent-based text classification models.
- Convolution/recurrent-based sentence embedding models.
- TorchText extensions for training (test/validation dataset split, word embeddings)
- Text/vision dataset loaders. (Omniglot, normal <-> simple wikipedia)
- Modular PyTorch model training utilities w/ model checkpoints, and validation loss/accuracy checks.
- How-to example PyTorch code snippets.