from torchtext import data
from torchtext.data import iterator
from torchtext.datasets import TranslationDataset

START_SENTENCE_TOKEN = "<s>"
END_SENTENCE_TOKEN = "</s>"


def simple_wikipedia():
    """
    "Simple English Wikipedia: A New Text Simplification Task"

    A text simplification dataset for sentence embeddings/text classification tasks.

    :return: Dataset w/ both normal and simple padded sentences.
    """
    text_field = data.Field(init_token=START_SENTENCE_TOKEN, eos_token=END_SENTENCE_TOKEN, pad_token=END_SENTENCE_TOKEN,
                            lower=True, include_lengths=True,
                            tokenize=lambda row: row.split('\t')[2].split())
    dataset = TranslationDataset('data/wikipedia/', ('normal.aligned', 'simple.aligned'),
                                 fields=[('normal', text_field), ('simple', text_field)],
                                 filter_pred=lambda example: example.normal != example.simple)
    text_field.build_vocab(dataset, wv_type='glove.6B')

    return dataset, text_field.vocab


if __name__ == "__main__":
    from nlp import utils

    dataset, vocab = simple_wikipedia()

    iterator = iterator.Iterator(dataset, 32, shuffle=True, device=-1)
    batch = next(iter(iterator))

    normal_sentences, normal_sentence_lengths = batch.normal
    normal_sentences = utils.embed_sentences(normal_sentences, vocab.vectors)

    print(normal_sentences.numpy())
