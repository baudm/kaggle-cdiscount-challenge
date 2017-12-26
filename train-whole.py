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
import threadutils
from buffering import buffered_gen_threaded as buf

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

#import functools

#buffered_gen_threaded = functools.partial(buffering.buffered_gen_threaded, buffer_size=4)
import json

with open('stats.json', 'r') as f:
    stats = json.load(f)

# @threadutils.threadsafe_generator
# @buffering.buffered_gen_threaded


def drop(features, categories):
    for_dropping = []
    for i, c in enumerate(categories):
        num_samples = stats[c]
        if np.random.uniform() > (1200./num_samples):
            for_dropping.append(i)
    #print('dropped:', len(for_dropping))
    features = np.delete(features, for_dropping, axis=0)
    categories = np.delete(categories, for_dropping, axis=0)
    return features, categories


def loader(paths, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = []
        for path in paths:
            files.extend(glob.glob(os.path.join(path, '*.npz')))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            features = archive['features']
            categories = archive['categories']
            del archive
            # print(features.shape)
            # features, categories = drop(features, categories)
            # print(features.shape)
            #features = features.reshape(256*256, -1)
            #categories = categories.flatten()
            shuffle_in_unison_scary(features, categories)
            # Split into mini batches
            num_batches = int(len(categories) / batch_size)
            #half = int(np.ceil(num_batches/2.))
            features = np.array_split(features, num_batches)
            categories = np.array_split(categories, num_batches)
            #can_preload = True
            while categories:
                batch_features = features.pop()
                batch_categories = categories.pop()
                # convert to one-hot representation
                #batch_categories = np.stack([to_categorical(c, num_classes) for c in batch_categories]).squeeze()
                yield batch_features, batch_categories
                # if can_preload and len(features) <= half and i + 1 < len(files):
                #     can_preload = False
                #     # preload next file
                #     np.load(files[i+1])['features']

def loader_test(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        #np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            features = archive['features']
            categories = archive['categories']
            del archive
            #features = features.reshape(256*256, -1)
            #categories = categories.flatten()
            shuffle_in_unison_scary(features, categories)
            # Split into mini batches
            num_batches = int(len(categories) / batch_size)
            #half = int(np.ceil(num_batches/2.))
            features = np.array_split(features, num_batches)
            categories = np.array_split(categories, num_batches)
            #can_preload = True
            while categories:
                batch_features = features.pop()
                batch_categories = categories.pop()
                # convert to one-hot representation
                #batch_categories = np.stack([to_categorical(c, num_classes) for c in batch_categories]).squeeze()
                yield batch_features, batch_categories
                # if can_preload and len(features) <= half and i + 1 < len(files):
                #     can_preload = False
                #     # preload next file
                #     np.load(files[i+1])['features']
            break

import queue
import os.path


npz_queue = queue.Queue(4)
batch_queue = queue.Queue(512)


def parloader(paths):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = []
        for path in paths:
            files.extend(glob.glob(os.path.join(path, '*.npz')))
        np.random.shuffle(files)
        for npz in files:
            npz_queue.put(npz)


def blah(batch_size):
    while True:
        npz = npz_queue.get()
        # Load pack into memory
        archive = np.load(npz)
        features = archive['features']
        categories = archive['categories']
        del archive
        # print(features.shape)
        #features, categories = drop(features, categories)
        # print(features.shape)
        # features = features.reshape(256*256, -1)
        # categories = categories.flatten()
        shuffle_in_unison_scary(features, categories)
        # Split into mini batches
        num_batches = int(len(categories) / batch_size)
        # half = int(np.ceil(num_batches/2.))
        features = np.array_split(features, num_batches)
        categories = np.array_split(categories, num_batches)
        # can_preload = True
        while categories:
            batch_features = features.pop()
            batch_categories = categories.pop()
            # convert to one-hot representation
            # batch_categories = np.stack([to_categorical(c, num_classes) for c in batch_categories]).squeeze()
            batch_queue.put((batch_features, batch_categories))


def batch_loader():
    while True:
        yield batch_queue.get()


from keras.callbacks import ModelCheckpoint

VAL_SET = 181597

import sys
from keras.models import load_model
from keras.layers import Conv1D, Reshape, Flatten, Conv2D

def make_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(2048,)))
    # model.add(Reshape((1, 1, 2048)))
    # model.add(Conv2D(2048, 1, activation='relu', padding='same'))
    # model.add(Conv2D(num_classes, 1, activation='softmax'))
    # model.add(Reshape((-1,)))

    #model.add(Flatten())
    #model.add(Dropout(0.2))
    # weight_decay = regularizers.l2(1e-5)

    #model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.1))

    # model.add(Dense(2048, activation='relu'))
    # model.add(Dense(4096, activation='relu', kernel_regularizer=weight_decay))
    # model.add(Dropout(0.1))
    # model.add(Dense(2048, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(3072, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler

from threading import Thread

from keras import regularizers
import json

w = {}
most = max(stats.values())
for k, v in stats.items():
    if v < 1200:
        v = 1200
    w[lookup_table[int(k)]] = most/v


from keras.applications import Xception
from keras.models import Model

import bson
from skimage.io import imread
import io


def train_loader(batch_size, normalize=True):
    images = []
    categories = []

    while True:
        with open('input/train.bson', 'rb') as f:
            data = bson.decode_file_iter(f)
            for d in data:
                cid = d['category_id']
                # Convert from category_id to index
                c = lookup_table[cid]
                #c = to_categorical(c, num_classes)
                ### Accelerate by using just 1 image per product
                for pic in d['imgs']:

                    # Randomly skip 3/4 of training samples
                    #if random.randint(0, 3) != 0:
                    #    continue

                    img = imread(io.BytesIO(pic['picture']))

                    images.append(img)
                    categories.append(c)


                    if len(categories) >= batch_size:
                        images = np.stack(images)
                        categories = np.stack(categories).squeeze()
                        if normalize:
                           images = (images - 127.5) / 127.5
                        # print(counter)
                        yield images, categories
                        images = []
                        categories = []


from buffering import buffered_gen_threaded as buf

def main(base_lr=0.1, batch_size=64):
    initial_epoch = 0
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        base_model = Xception(include_top=False, weights=None, input_shape=(180, 180, 3), pooling='avg')
        model = load_model(fname, compile=False)
        combined = Model(base_model.input, model(base_model.output))
        for layer in base_model.layers:
            if not layer.name.startswith('block14'):
                layer.trainable = False
        model = combined
        try:
            initial_epoch = int(fname.split('.')[1]) + 1
        except (IndexError, ValueError):
            pass
        # for layer in model.layers:
        #     if isinstance(layer, Dense):
        #         layer.kernel_regularizer = regularizers.l2(1e-5)
        #     elif layer.name == 'dropout_1':
        #         layer.rate = 0.2
        #     elif layer.name == 'dropout_2':
        #         layer.rate = 0.3
        #     print(layer.get_config())
            # if isinstance(layer, Dropout):# and layer.name in ['dropout_1', 'dropout_2', 'dropout_4']:
            #     layer.rate = 0.5
            #     print('set dropout rate to zero', layer)

    else:
        model = make_model()

    #print(model.get_config())
    model.summary()


    k = 64/2
    base_lr = 0.05


    sgd = SGD(base_lr, momentum=0.9, nesterov=True)

    rmsprop = RMSprop()

    model.compile('adadelta', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    num_samples = 12371293 - VAL_SET + 62*256*256 + 53662
    batch_size = 256
    steps_per_epoch = int(np.ceil(num_samples/batch_size))
    epochs = 100

    num_workers = 1

    def schedule(epoch):
        e = int(epoch /10)
        #print('decay', e)
        return base_lr * (0.94**e)

    lr_scheduler = LearningRateScheduler(schedule)
    checkpoint = ModelCheckpoint('/home/darwin/Projects/kaggle/cdiscount/model.{epoch:02d}.hdf5', verbose=True, save_best_only=True)

    buf_size = 10*256/batch_size

    dirs = ['/var/local/features', '/home/darwin/Projects/kaggle/cdiscount/features']


    ldr = buf(train_loader(batch_size), buf_size)
    return model.fit_generator(ldr, steps_per_epoch, epochs, callbacks=[checkpoint]
                        , validation_data=loader(['/var/local/features/val'], batch_size), validation_steps=int(VAL_SET/batch_size)
                        , workers=num_workers,
                               # class_weight=w,
                        initial_epoch=initial_epoch
                        )

import tensorflow as tf
from keras import backend as K

import itertools

if __name__ == '__main__':
    main()
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)
    # K.set_session(session)
    #
    # lrs = [0.1]
    # bs = [256]
    #
    # best_loss = 100
    # best_bs = -1
    # best_lr = -1
    #
    # tries = 5
    #
    # for batch_size, lr in itertools.product(bs, lrs):
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     session = tf.Session(config=config)
    #     K.set_session(session)
    #     print(lr, batch_size)
    #     loss = 0
    #
    #     for i in range(tries):
    #         h = main(lr, batch_size)
    #         loss += h.history['val_loss'][-1]
    #
    #     loss /= tries
    #
    #     if loss < best_loss:
    #         best_loss = loss
    #         best_bs = batch_size
    #         best_lr = lr
    #
    #         print(best_bs, best_lr, best_loss)
    #
    # # best: 32, 0.01
    # # 64, 0.1
    # # 32, 0.001
    # # 256 0.1 5.22753078534
    #
    # print(best_bs, best_lr, best_loss)
