#!/usr/bin/env python

import glob
import os.path
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer

import pickle

with open('category-table', 'rb') as f:
    lookup_table = pickle.load(f)
num_classes = len(lookup_table)

from keras.utils import to_categorical

def loader(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        for npz in glob.iglob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            features = archive['features']
            categories = archive['categories']
            del archive
            #features = features.reshape(256*256, -1)
            #categories = categories.flatten()
            # Split into mini batches
            num_batches = int(len(categories) / batch_size)
            features = np.array_split(features, num_batches)
            categories = np.array_split(categories, num_batches)
            while categories:
                batch_features = features.pop()
                batch_categories = categories.pop()
                # convert to one-hot representation
                batch_categories = [to_categorical(c, num_classes) for c in batch_categories]
                yield batch_features, batch_categories


from keras.callbacks import ModelCheckpoint


def main():
    model = Sequential()
    model.add(InputLayer((2048, )))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    num_samples = 12371293
    batch_size = 512
    steps_per_epoch = 1000
    epochs = int(np.round(num_samples / (batch_size * steps_per_epoch)))

    checkpoint = ModelCheckpoint('/home/darwin/Projects/kaggle/cdiscount/model.{epoch:02d}.hdf5')

    model.fit_generator(loader('features', batch_size), steps_per_epoch, epochs, callbacks=[checkpoint]
                        , validation_data=loader('.', batch_size), validation_steps=20)


if __name__ == '__main__':
    main()