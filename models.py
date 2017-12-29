#!/usr/bin/env python

import os.path

from keras.applications import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, InputLayer
from keras.utils.data_utils import get_file


from data import NUM_CLASSES


WEIGHTS_PATH = 'https://github.com/baudm/kaggle-cdiscount-challenge/raw/master/weights/classifier_weights_tf_dim_ordering_tf_kernels.h5'


def create_classifier_model(use_weights=False):
    classifier = Sequential()
    classifier.add(InputLayer(input_shape=(2048,)))
    classifier.add(Dense(4096, activation='relu'))
    classifier.add(Dense(4096, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(NUM_CLASSES, activation='softmax'))

    if use_weights:
        weights_name = os.path.basename(WEIGHTS_PATH)
        weights_path = get_file(weights_name, WEIGHTS_PATH, cache_dir='.', cache_subdir='weights',
                                file_hash='0fa9d0d2fe9574a1f588ff04b12c22d036dd958d6e4f697f7ec59c873c942d2b')
        classifier.load_weights(weights_path)

    return classifier


def create_full_model(top_model=None):
    base_model = Xception(include_top=False, input_shape=(180, 180, 3), pooling='avg')
    if top_model is None:
        top_model = create_classifier_model(True)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    return model
