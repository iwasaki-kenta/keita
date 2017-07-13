
def validation_split(examples, split_factor=0.7):
    cut_index = int(len(examples) * split_factor)

    train_examples = examples[:cut_index]
    validation_examples = examples[cut_index:]

    return train_examples, validation_examples