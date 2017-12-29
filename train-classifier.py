#!/usr/bin/env python

import sys

import numpy as np

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD

from buffering import buffered_gen_threaded as buffered
from models import create_classifier_model
import data


def main():
    initial_epoch = 0
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        model = load_model(fname, compile=False)
        try:
            initial_epoch = int(fname.split('.')[1]) + 1
        except (IndexError, ValueError):
            pass
    else:
        model = create_classifier_model()

    model.summary()

    base_lr = 0.01
    sgd = SGD(base_lr, momentum=0.9, nesterov=True)

    model.compile(sgd, 'sparse_categorical_crossentropy', metrics=['accuracy'])

    train_samples = data.NUM_SAMPLES - data.VAL_SET
    batch_size = 256
    steps_per_epoch = int(np.ceil(train_samples / batch_size))
    epochs = 100

    def schedule(epoch):
        return base_lr * (0.98 ** epoch)

    lr_scheduler = LearningRateScheduler(schedule)
    checkpoint = ModelCheckpoint('classifier.{epoch:02d}.h5', verbose=True, save_best_only=True)

    buf_size = 2*256*256/batch_size

    dirs = ['/var/local/features']

    train_loader = buffered(data.feature_loader(dirs, batch_size), buf_size)
    val_loader = buffered(data.feature_loader(['/var/local/features/val'], batch_size), buf_size)
    model.fit_generator(train_loader, steps_per_epoch, epochs, callbacks=[checkpoint, lr_scheduler],
                        validation_data=val_loader, validation_steps=data.VAL_SET//batch_size,
                        initial_epoch=initial_epoch
                        )


if __name__ == '__main__':
    main()
