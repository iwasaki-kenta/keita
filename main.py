"""
An example of how to use the utilities in Keita.
"""

if __name__ == "__main__":
    from nlp.models.rnn import BidirectionalEncoder
    from nlp import utils
    from datasets import nlp
    from torchtext import data
    from torch import nn, optim, autograd
    from tqdm import tqdm
    import torch

    batch_size = 32
    embed_size = 300

    model = BidirectionalEncoder(embed_dim=embed_size, hidden_dim=512, num_layers=1)
    if torch.cuda.is_available(): model = model.cuda()

    train, valid, vocab = nlp.simple_wikipedia(split_factor=0.9)
    vocab.vectors = vocab.vectors.cpu()

    padding_token = vocab.vectors[vocab.stoi[nlp.PADDING_TOKEN]]

    train_iterator = data.iterator.Iterator(train, batch_size, shuffle=True, device=-1)
    valid_iterator = data.iterator.Iterator(valid, batch_size, shuffle=True, device=-1)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        average_training_loss, average_validation_loss = 0, 0
        num_samples = 0

        model = model.train()
        for valid_batch in tqdm(train_iterator):
            normal_sentences, normal_sentence_lengths = valid_batch.normal
            simple_sentences, simple_sentence_lengths = valid_batch.simple

            normal_sentences = utils.embed_sentences(normal_sentences, vocab.vectors)
            simple_sentences = utils.embed_sentences(simple_sentences, vocab.vectors)

            sentences = utils.concat_sentence_batches(normal_sentences, simple_sentences, padding_token)
            sentence_lengths = torch.cat([normal_sentence_lengths, simple_sentence_lengths], dim=0)
            labels = torch.LongTensor([0] * batch_size + [1] * batch_size)

            # Shuffle the batch around.
            random_indices = torch.randperm(sentences.size(1))

            sentences = sentences.index_select(1, random_indices)
            sentence_lengths = sentence_lengths[random_indices]
            labels = labels[random_indices]

            if torch.cuda.is_available():
                sentence_lengths = sentence_lengths.cuda()
                sentences = sentences.cuda()
                labels = labels.cuda()

            sentences = autograd.Variable(sentences)
            sentence_lengths = autograd.Variable(sentence_lengths)
            labels = autograd.Variable(labels)

            optimizer.zero_grad()
            outputs = model((sentences, sentence_lengths))

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # Make logs.
            average_training_loss += loss.data[0]
            num_samples += 1

        average_training_loss /= num_samples
        print("Epoch %d - Loss: %f" % (epoch, average_training_loss))

        model = model.eval()
        for valid_batch in tqdm(valid_iterator):
            normal_sentences, normal_sentence_lengths = valid_batch.normal
            simple_sentences, simple_sentence_lengths = valid_batch.simple

            normal_sentences = utils.embed_sentences(normal_sentences, vocab.vectors)
            simple_sentences = utils.embed_sentences(simple_sentences, vocab.vectors)

            sentences = utils.concat_sentence_batches(normal_sentences, simple_sentences, padding_token)
            sentence_lengths = torch.cat([normal_sentence_lengths, simple_sentence_lengths], dim=0)
            labels = torch.LongTensor([0] * batch_size + [1] * batch_size)

            # Shuffle the batch around.
            random_indices = torch.randperm(sentences.size(1))

            sentences = sentences.index_select(1, random_indices)
            sentence_lengths = sentence_lengths[random_indices]
            labels = labels[random_indices]

            if torch.cuda.is_available():
                sentence_lengths = sentence_lengths.cuda()
                sentences = sentences.cuda()
                labels = labels.cuda()

            sentences = autograd.Variable(sentences)
            sentence_lengths = autograd.Variable(sentence_lengths)
            labels = autograd.Variable(labels)
            outputs = model((sentences, sentence_lengths))

            loss = criterion(outputs, labels)

            # Make logs.
            average_validation_loss += loss.data[0]
            num_samples += 1

        average_validation_loss /= num_samples
        print("Validation - Loss: %f" % (average_validation_loss))
