if __name__ == "__main__":
    import torch
    import random

    """
    Testing how to sort/unsort in PyTorch.
    """

    sentences = []
    for x in range(10): sentences.append(torch.FloatTensor([random.random() for _ in range(random.randint(4, 16))]))

    sentence_lengths = torch.LongTensor([len(sentence) for sentence in sentences])
    max_sentence_length = max(sentence_lengths)

    sentence_batch = torch.zeros(10, max_sentence_length + 1, 1)
    for index, sentence in enumerate(sentences):
        sentence_batch[index][:len(sentence)] = sentence

    print(sentence_batch)

    sorted_sentence_lengths, sort_indices = torch.sort(sentence_lengths, dim=0, descending=True)
    sorted_sentences = sentence_batch[sort_indices]

    print(sentence_lengths.numpy())
    print(sorted_sentence_lengths.numpy())

    _, unsort_indices = torch.sort(sort_indices, dim=0)
    unsorted_sentences = sorted_sentences[unsort_indices]

    zeros = torch.nonzero(unsorted_sentences == 0).numpy()
    lengths = {}
    for pair in zeros:
        if pair[0] not in lengths:
            lengths[pair[0]] = pair[1]
    print(list(lengths.values()))