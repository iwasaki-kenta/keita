from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import iterator

START_SENTENCE_TOKEN = "<s>"
END_SENTENCE_TOKEN = "</s>"


def simple_wikipedia():
    """
    "Simple English Wikipedia: A New Text Simplification Task"

    Provided for the sake of creating sentence embeddings/text classification tasks.

    :return: Dataset w/ both normal and simple padded sentences.
    """
    text_field = data.Field(init_token=START_SENTENCE_TOKEN, eos_token=END_SENTENCE_TOKEN, pad_token=END_SENTENCE_TOKEN,
                            lower=True, include_lengths=True,
                            tokenize=lambda row: row.split('\t')[2].split())
    dataset = TranslationDataset('data/wikipedia/', ('normal.aligned', 'simple.aligned'),
                                 fields=[('normal', text_field), ('simple', text_field)])
    text_field.build_vocab(dataset)

    return dataset


if __name__ == "__main__":
    dataset = simple_wikipedia()

    iterator = iterator.Iterator(dataset, 32, shuffle=True)
    batch = next(iter(iterator))

    print(batch.normal)
    print(batch.simple)
