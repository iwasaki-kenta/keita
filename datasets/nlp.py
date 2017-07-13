from torchtext import data
from torchtext.data.iterator import Iterator
from datasets.utils import validation_split

import os

START_SENTENCE_TOKEN = "<s>"
END_SENTENCE_TOKEN = "</s>"


def simple_wikipedia(split_factor=0.7):
    """
    "Simple English Wikipedia: A New Text Simplification Task"

    A text simplification dataset for sentence embeddings/text classification tasks.

    :return: Dataset w/ both normal and simple padded sentences.
    """

    text_field = data.Field(init_token=START_SENTENCE_TOKEN, eos_token=END_SENTENCE_TOKEN, pad_token=END_SENTENCE_TOKEN,
                            lower=True, include_lengths=True,
                            tokenize=lambda row: row.split('\t')[2].split())
    fields = [('normal', text_field), ('simple', text_field)]

    source_path, target_path = tuple(
        os.path.expanduser('data/wikipedia/' + x) for x in ['normal.aligned', 'simple.aligned'])

    examples = []
    with open(source_path) as source_file, open(target_path) as target_file:
        for normal_line, target_line in zip(source_file, target_file):
            normal_line, target_line = normal_line.strip(), target_line.strip()
            if normal_line != '' and target_line != '':
                examples.append(data.Example.fromlist([normal_line, target_line], fields))

    train_examples, validation_examples = validation_split(examples, split_factor)

    equality = lambda example: example.normal != example.simple

    train_dataset = data.Dataset(train_examples, fields, filter_pred=equality)
    validation_dataset = data.Dataset(validation_examples, fields, filter_pred=equality)

    text_field.build_vocab(train_dataset, validation_dataset, wv_type='glove.6B')

    return train_dataset, validation_dataset, text_field.vocab


if __name__ == "__main__":
    from nlp import utils

    train, valid, vocab = simple_wikipedia()

    train_iterator = Iterator(train, 32, shuffle=True, device=-1)
    valid_iterator = Iterator(valid, 32, shuffle=True, device=-1)

    train_batch = next(iter(train_iterator))
    valid_batch = next(iter(valid_iterator))

    normal_sentences, normal_sentence_lengths = train_batch.normal
    normal_sentences = utils.embed_sentences(normal_sentences, vocab.vectors)

    print("A normal batch looks like %s. " % str(normal_sentences.size()))
    print("The dataset contains %d train samples, %d validation samples w/ a vocabulary size of %d. " % (
        len(train), len(valid), len(vocab)))
