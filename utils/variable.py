"""
Some utilities for ensuring that tensor data is on the GPU/CPU.
"""

from torch import autograd
import torch


def variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    var = autograd.Variable(tensor)
    return var


def variables(*tensors):
    return [variable(tensor) for tensor in tensors]
