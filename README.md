# Keita: A PyTorch Toolkit

## Description

A couple of PyTorch utilities, dataset loaders, and layers suitable for natural language processing, computer vision, meta-learning, etc. which I'm opening out to the community.

I cannot guarantee fixing potential bugs you may find whatsoever; though if you'd like to report any then feel free to file an issue/pull request and I'll try my luck on it. Feedback and suggestions are definitely appreciated!

In terms of code organization, I would like to clarify that I myself am not a fan of using huge repositories of highly un-maintained, dependant code and thus intend to keep this repository as modular as possible. Hence, for all modules you wish to use in your project, copy-pasting the module alongside a few utility methods should be all that you need to do to get it incorporated into your project.

I intend to make the code as clean and well-documented as possible by keeping the code style consistent and developer-friendly (clear variable names, simple references to different modules within the toolkit, etc.).

## Dependencies

PyTorch, TorchVision, TQDM, and the bleeding edge build version of TorchText required if you wish to use all the modules within this toolkit.

## Contents

- Deep metric learning losses. (mahalonobis-distance hard negative mining)
- Probabilistic/non-linear models. (gaussian mixture models, conditional random fields)
- Meta-learning models. (temporal convolution meta-learner)
- Activation unit layers. (gated activation unit for PixelCNN)
- Extended convolution layer support. (separable convolutions, causal convolutions)
- Convolution/recurrent-based inter-attention layers (additive, dot-product, concat, bidirectional, bilinear)
- Convolution/recurrent-based text classification models.
- Convolution/recurrent-based sentence embedding models.
- TorchText extensions for training (test/validation dataset split, word embeddings)
- Text/vision dataset loaders. (Omniglot, normal <-> simple wikipedia)
- Modular PyTorch model training utilities w/ model checkpoints, and validation loss/accuracy checks.
- How-to example PyTorch code snippets.

## Papers I've Implemented w/ Keita

- A Deep Reinforced Model for Abstractive Summarization
- Meta-Learning with Temporal Convolutions
- Conditional Image Generation with PixelCNN Decoders
- WaveNet: A Generative Model for Raw Audio
- Deep Metric Learning via Lifted Structured Feature Embedding
- Max-Margin Object Detection
- Neural Machine Translation by Jointly Learning to Align and Translate
- Effective Approaches to Attention-based Neural Machine Translation
- DeXpression: Deep Convolutional Neural Network for Expression Recognition
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- YOLO9000: Better, Faster, Stronger
- A Deep Reinforced Model for Abstractive Summarization
- Bidirectional LSTM-CRF Models for Sequence Tagging
- Discriminative Deep Metric Learning for Face Verification in the Wild
- Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
- A Neural Representation of Sketch Drawings
- Hierarchical Attention Networks for Document Classification

## Example Snippets

```python
"""
Create a PyTorch trainer which handles model checkpointing/loss/accuracy tracking given
training and validation dataset iterators.
"""

from text.models import classifiers
from text.models.cnn import encoders
from datasets import text
from torchtext import data
from torch import nn, optim
from train.utils import train_epoch, TrainingProgress
import torch

batch_size = 32
embed_size = 300

model = classifiers.LinearNet(embed_dim=embed_size, hidden_dim=64,
                              encoder=encoders.HierarchialNetwork1D,
                              num_classes=2)
if torch.cuda.is_available(): model = model.cuda()

train, valid, vocab = text.simple_wikipedia(split_factor=0.9)
vocab.vectors = vocab.vectors.cpu()

sort_key = lambda batch: data.interleave_keys(len(batch.normal), len(batch.simple))
train_iterator = data.iterator.Iterator(train, batch_size, shuffle=True, device=-1, repeat=False, sort_key=sort_key)
valid_iterator = data.iterator.Iterator(valid, batch_size, device=-1, train=False, sort_key=sort_key)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

progress = TrainingProgress()

def training_process(batch, train):
    # Process batch here and return torch.autograd.Variable's representing loss and accuracy.
    return loss, acc

for epoch in range(100):
        train_epoch(epoch, model, train_iterator, valid_iterator, processor=training_process, progress=progress)
```

```python
"""
Load a text dataset padded, embedded w/ GloVe word vectors, sorted according to sentence length
for direct use with PyTorch's pad packing for RNN modules and print some statistics.
"""

from text import utils
from torchtext.data.iterator import Iterator
from datasets.text import simple_wikipedia
from torchtext import data

train, valid, vocab = simple_wikipedia()

sort_key = lambda batch: data.interleave_keys(len(batch.normal), len(batch.simple))
train_iterator = Iterator(train, 32, shuffle=True, device=-1, repeat=False, sort_key=sort_key)
valid_iterator = Iterator(valid, 32, device=-1, train=False, sort_key=sort_key)

train_batch = next(iter(train_iterator))
valid_batch = next(iter(valid_iterator))

normal_sentences, normal_sentence_lengths = train_batch.normal
normal_sentences = utils.embed_sentences(normal_sentences, vocab.vectors)

print("A normal batch looks like %s. " % str(normal_sentences.size()))
print("The dataset contains %d train samples, %d validation samples w/ a vocabulary size of %d. " % (
    len(train), len(valid), len(vocab)))
```

```python
"""
Paulus et al. encoder/decoder attention layer example usage for the paper
"A Deep Reinforced Model for Abstractive Summarization"

https://arxiv.org/abs/1705.04304
"""

from layers.attention import BilinearAttention
import torch

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
```

```python
"""
1D dilated causal convolutions for models like WaveNet and the Temporal Convolution Meta-Learner (TCML).

WaveNet: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
TCML: https://arxiv.org/abs/1707.03141
"""

from layers.convolution import CausalConv1d
import torch

image = torch.arange(0, 4).unsqueeze(0).unsqueeze(0)
image = torch.autograd.Variable(image)

layer = CausalConv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=1)
layer.weight.data.fill_(1)
layer.bias.data.fill_(0)

print(image.data.numpy())
print(layer(image).round().data.numpy())
```
