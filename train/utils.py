import torch
from tqdm import tqdm


class TrainingProgress:
    average_training_loss, average_validation_loss = 0, 0
    average_training_acc, average_validation_acc = 0, 0

    def __init__(self, epoch=0, track_accuracy=True, best_validation_loss=1e6):
        self.epoch = epoch
        self.track_accuracy = track_accuracy
        self.best_validation_loss = best_validation_loss

    def update_progress(self, train, epoch, loss, acc):
        self.epoch = epoch

        if train:
            self.average_training_loss += loss

            if self.track_accuracy:
                self.average_training_acc += acc
        else:
            self.average_validation_loss += loss

            if self.track_accuracy:
                self.average_validation_acc += acc

    def finish_epoch(self, train, epoch, model, num_batches):
        self.epoch = epoch

        if train:
            self.average_training_acc /= num_batches
            self.average_training_loss /= num_batches
        else:
            self.average_validation_acc /= num_batches
            self.average_validation_loss /= num_batches

            # Model checkpoint.
            if self.best_validation_loss > self.average_validation_loss:
                self.best_validation_loss = self.average_validation_loss
                torch.save(model.state_dict(), "epoch-%d-%d.pt" % (self.epoch, int(self.average_validation_acc)))


def train_epoch(epoch, model, train_iterator, valid_iterator, processor, progress):
    """
    Trains one epoch for a given model given train and validation data-sets.
    Saves model checkpoints based on validation loss, and optionally logs accuracy.

    :param best_validation_loss: Current best validation loss.
    :param epoch: Current epoch index.
    :param model: Model to be trained.
    :param train_iterator: Dataset iterator for train dataset.
    :param valid_iterator: Dataset iterator for validation dataset.
    :param processor: A function that processes one batch
    :param accuracy: Boolean to signify whether or not to log accuracy.
    """
    assert progress is not None, "Please provide your training progress to train_epoch."

    num_batches = 0
    model = model.train()
    for batch in tqdm(train_iterator):
        if progress.track_accuracy:
            training_loss, training_acc = processor(batch)
        else:
            training_loss = processor(batch)

        progress.update_progress(epoch=epoch, train=True, loss=training_loss.data[0], acc=training_acc.data[0])
        num_batches += 1

    progress.finish_epoch(train=True, epoch=epoch, model=model, num_batches=num_batches)

    print("Epoch %d - Loss: %f - Accuracy: %.2f%%" % (
        epoch, progress.average_training_loss, progress.average_training_acc))

    num_batches = 0
    model = model.eval()
    for batch in tqdm(valid_iterator):
        if progress.track_accuracy:
            valid_loss, valid_acc = processor(batch)
        else:
            valid_loss = processor(batch)

        progress.update_progress(epoch=epoch, train=False, loss=valid_loss.data[0], acc=valid_acc.data[0])
        num_batches += 1

    progress.finish_epoch(train=False, epoch=epoch, model=model, num_batches=num_batches)

    print("Validation - Loss: %f - Accuracy: %.2f%%" % (
        progress.average_validation_loss, progress.average_validation_acc))
