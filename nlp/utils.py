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
