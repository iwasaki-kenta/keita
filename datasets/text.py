import os
import re

from torchtext import data

from datasets.utils import validation_split

START_SENTENCE_TOKEN = "<s>"
END_SENTENCE_TOKEN = "</s>"
PADDING_TOKEN = "<p>"

DATA_DIRECTORY = "data"


def simple_wikipedia(split_factor=0.7, word_vectors='glove.6B'):
    """
    "Simple English Wikipedia: A New Text Simplification Task"

    A text simplification dataset for sentence embeddings/text classification tasks.

    :return: Dataset w/ both normal and simple padded sentences.
    """

    text_field = data.Field(init_token=START_SENTENCE_TOKEN, eos_token=END_SENTENCE_TOKEN, pad_token=PADDING_TOKEN,
                            lower=True, include_lengths=True,
                            tokenize=lambda row: row.split('\t')[2].split(' '))
    fields = [('normal', text_field), ('simple', text_field)]

    source_path, target_path = tuple(
        os.path.expanduser(DATA_DIRECTORY + '/wikipedia/' + x) for x in ['normal.aligned', 'simple.aligned'])

    examples = []
    with open(source_path, errors='replace') as source_file, open(target_path, errors='replace') as target_file:
        for normal_line, target_line in zip(source_file, target_file):
            normal_line, target_line = normal_line.strip(), target_line.strip()
            if normal_line != '' and target_line != '':
                examples.append(data.Example.fromlist([normal_line, target_line], fields))

    train_examples, validation_examples = validation_split(examples, split_factor)

    equality = lambda example: example.normal != example.simple

    train_dataset = data.Dataset(train_examples, fields, filter_pred=equality)
    validation_dataset = data.Dataset(validation_examples, fields, filter_pred=equality)

    text_field.build_vocab(train_dataset, validation_dataset, wv_dir=DATA_DIRECTORY, wv_type=word_vectors)

    return train_dataset, validation_dataset, text_field.vocab


def bAbI(word_vectors='glove.6B'):
    context_field = data.Field(init_token=START_SENTENCE_TOKEN, eos_token=END_SENTENCE_TOKEN, pad_token=PADDING_TOKEN,
                               lower=True, include_lengths=True, tokenize=lambda x: x)
    answer_field = data.Field(lower=True, sequential=False)
    support_field = data.Field(init_token=START_SENTENCE_TOKEN, eos_token=END_SENTENCE_TOKEN, pad_token=PADDING_TOKEN,
                               tokenize=lambda x: x)
    fields = [('context', context_field), ('question', context_field), ('answers', answer_field),
              ('support', support_field)]

    train_examples, test_examples = [], []

    def tokenize(sentence):
        return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]

    def decode_story(file):
        with open(os.path.join(root, file), errors='replace') as data_file:
            story = []

            for line in data_file:
                new_index, line = line.strip().split(' ', 1)
                new_index = int(new_index)

                if new_index == 1:
                    story = []
                if '\t' in line:
                    question, answer, supporting = line.split('\t')
                    question = tokenize(question)
                    supporting = [index - 1 for index in map(int, supporting.split(' '))]

                    sentences = [word for sentence in story for word in sentence if sentence]

                    story_map = {}
                    for index, sentence in enumerate(story):
                        story_map[index] = sentence

                    sequence = []
                    for key in sorted(list(story_map.keys())):
                        if not story_map[key]:
                            continue

                        if key in supporting:
                            # Omit last character (usually a period . )
                            assert len((['B'] + ['I'] * (len(story_map[key]) - 2) + ['O'])) == len(
                                story_map[key]), 'Something went wrong.'
                            sequence += (['B'] + ['I'] * (len(story_map[key]) - 2) + ['O'])
                        else:
                            sequence += ['O'] * len(story_map[key])

                    assert len(sequence) == len(sentences), "POS tags aren't the same length as sentences?"

                    details = [sentences, question, answer, sequence]

                    if file.endswith('_train.txt'):
                        train_examples.append(data.Example.fromlist(details, fields))
                    else:
                        test_examples.append(data.Example.fromlist(details, fields))

                    story.append('')
                else:
                    sentence = tokenize(line)
                    story.append(sentence)

    for root, dirs, files in os.walk(DATA_DIRECTORY + '/bAbI/'):
        for file in files:
            if file.endswith('.txt'):
                decode_story(file)

    train_dataset = data.Dataset(train_examples, fields)
    test_dataset = data.Dataset(test_examples, fields)

    context_field.build_vocab(train_dataset, test_dataset, wv_dir=DATA_DIRECTORY, wv_type=word_vectors)
    answer_field.build_vocab(train_dataset, test_dataset, wv_dir=DATA_DIRECTORY, wv_type=word_vectors)
    support_field.build_vocab(train_dataset, test_dataset)

    return train_dataset, test_dataset, context_field.vocab, answer_field.vocab, support_field.vocab


if __name__ == "__main__":
    from torchtext.data.iterator import Iterator

    train, test, text_vocab, answer_vocab, tag_vocab = bAbI()

    sort_key = lambda batch: data.interleave_keys(len(batch.context), len(batch.question))
    train_iterator = Iterator(train, 1, shuffle=True, device=-1, repeat=False, sort_key=sort_key)
    valid_iterator = Iterator(test, 1, device=-1, train=False, sort_key=sort_key)

    train_batch = next(iter(train_iterator))
    valid_batch = next(iter(valid_iterator))

    normal_sentences, normal_sentence_lengths = train_batch.context
    # normal_sentences = utils.embed_sentences(normal_sentences, text_vocab.vectors)

    print(normal_sentences)
    print(train_batch.support)

    print(text_vocab.stoi)
    print(answer_vocab.stoi)
    print(tag_vocab.stoi)

    print("A normal batch looks like %s. " % str(normal_sentences.size()))
    print(
        "The dataset contains %d train samples, %d validation samples w/ a text vocabulary size of %d, and an answers vocabulary size of %d. " % (
            len(train), len(test), len(text_vocab), len(answer_vocab)))
