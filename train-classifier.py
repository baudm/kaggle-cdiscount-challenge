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
            half = int(np.ceil(num_batches/2.))
            features = np.array_split(features, num_batches)
            categories = np.array_split(categories, num_batches)
            can_preload = True
            while categories:
                batch_features = features.pop()
                batch_categories = categories.pop()
                # convert to one-hot representation
                batch_categories = np.stack([to_categorical(c, num_classes) for c in batch_categories]).squeeze()
                yield batch_features, batch_categories
                # if can_preload and len(features) <= half and i + 1 < len(files):
                #     can_preload = False
                #     # preload next file
                #     np.load(files[i+1])['features']


from keras.callbacks import ModelCheckpoint

VAL_SET = 242*512

import sys
from keras.models import load_model

def make_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(2048,)))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def main():
    if len(sys.argv) == 2:
        model = load_model(sys.argv[1], compile=False)
    else:
        model = make_model()

    model.summary()

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    num_samples = 12371293 - VAL_SET
    batch_size = 512*4*2
    steps_per_epoch = int(np.ceil(num_samples/batch_size))
    epochs = 50

    checkpoint = ModelCheckpoint('/home/darwin/Projects/kaggle/cdiscount/model.{epoch:02d}.hdf5')

    model.fit_generator(loader('features', batch_size), steps_per_epoch, epochs, callbacks=[checkpoint]
                        , validation_data=loader('.', batch_size), validation_steps=int(VAL_SET/batch_size)
                        )


if __name__ == '__main__':
    main()