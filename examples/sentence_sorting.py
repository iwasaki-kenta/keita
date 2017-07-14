if __name__ == "__main__":
    import torch
    from text.utils import test_sentences

    """
    Testing how to sort/unsort batches of sentences w/ padding in PyTorch.
    """
    sentence_batch, sentence_lengths = test_sentences()

    print(sentence_batch.numpy())

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