from tqdm import tqdm
import torch


def train_epoch(epoch, model, train_iterator, valid_iterator, processor, accuracy=True):
    """
    Trains one epoch for a given model given training and validation data-sets.
    Saves model checkpoints based on validation loss, and optionally logs accuracy.

    :param epoch: Current epoch index.
    :param model: Model to be trained.
    :param train_iterator: Dataset iterator for training dataset.
    :param valid_iterator: Dataset iterator for validation dataset.
    :param processor: A function that processes one batch
    :param accuracy: Boolean to signify whether or not to log accuracy.
    """

    best_validation_loss = 1e6
    average_training_loss, average_validation_loss = 0, 0
    average_training_acc, average_validation_acc = 0, 0

    num_samples = 0
    model = model.train()
    for batch in tqdm(train_iterator):
        if accuracy:
            training_loss, training_acc = processor(batch)
        else:
            training_loss = processor(batch)

        average_training_loss += training_loss.data[0]
        if accuracy:
            average_training_acc += training_acc.data[0]
        num_samples += 1

    average_training_loss /= num_samples
    average_training_acc /= num_samples

    print("Epoch %d - Loss: %f - Accuracy: %.2f%%" % (epoch, average_training_loss, average_training_acc))

    num_samples = 0
    model = model.eval()
    for batch in tqdm(valid_iterator):
        if accuracy:
            valid_loss, valid_acc = processor(batch)
        else:
            valid_loss = processor(batch)

        average_validation_loss += valid_loss.data[0]
        if accuracy:
            average_validation_acc += valid_acc.data[0]
        num_samples += 1

    average_validation_loss /= num_samples
    average_validation_acc /= num_samples

    print("Validation - Loss: %f - Accuracy: %.2f%%" % (average_validation_loss, average_validation_acc))

    # Model checkpoint.
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        torch.save(model.state_dict(), "epoch-%d-%d.pt" % (epoch, int(average_validation_acc)))
