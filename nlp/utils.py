import random
import torch


def test_sentences(num_sentences=10, min_length=4, max_length=16):
    sentences = []
    for x in range(num_sentences):
        sentences.append(torch.FloatTensor([random.random() for _ in range(random.randint(min_length, max_length))]))

    sentence_lengths = torch.LongTensor([len(sentence) for sentence in sentences])
    max_sentence_length = max(sentence_lengths)

    # Add 1 to the max sentence length to designate end of sentence token.
    sentence_batch = torch.zeros(num_sentences, max_sentence_length + 1)
    for index, sentence in enumerate(sentences):
        sentence_batch[index][:len(sentence)] = sentence

    return sentence_batch, sentence_lengths


def embed_sentences(sentences, word_vectors):
    """
    Applies word embeddings to a batch of sentences.

    :param sentences: torch.LongTensor w/ shape [seq. length, batch size]
    :param word_vectors: torch.FloatTensor w/ shape [word indices, embed. dim]
    :return: Embeddings for a given batch of sentence [seq. length, batch size, embed. dim].
    """

    batch_size = sentences.size(1)
    sentences = word_vectors.index_select(dim=0, index=sentences.data.view(-1).long())
    return sentences.view(-1, batch_size, word_vectors.size(1))


def concat_sentence_batches(source_batch, target_batch, padding_token=None):
    """
    Concatenates two batches of sentences together, and fixes their padding accordingly.

    :param source_batch: First sentence batch. (seq. length, batch size, embed. dim)
    :param target_batch: Second sentence batch. (seq. length, batch size, embed. dim)
    :param padding_token: A padding token to be used for concatenation of the batches.
    :return: Concatenated sentence batch. (max(|source seq.|, |target seq.|), |source| + |target|, embed. dim)
    """
    if padding_token is None:
        padding_token = torch.LongTensor([0])
    source_padding_token, target_padding_token = padding_token, padding_token
    source_padding_token = source_padding_token.expand(source_batch.size(1), source_batch.size(2))
    target_padding_token = target_padding_token.expand(target_batch.size(1), target_batch.size(2))

    sentences = torch.zeros(max(source_batch.size(0), target_batch.size(0)),
                            source_batch.size(1) + target_batch.size(1), source_batch.size(2))
    for index in range(sentences.size(0)):
        if index < len(source_batch):
            sentences[index][:source_batch.size(1)] = source_batch[index]
        else:
            sentences[index][:source_batch.size(1)] = source_padding_token
        if index < len(target_batch):
            sentences[index][source_batch.size(1):] = target_batch[index]
        else:
            sentences[index][source_batch.size(1):] = target_padding_token

    return sentences
