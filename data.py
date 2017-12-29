#!/usr/bin/env python

import json
import bson
from skimage.io import imread
import io

import glob
import os.path

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


NUM_SAMPLES = 12371293

VAL_NUM = 10
VAL_SET = 256*256*VAL_NUM


with open('category-to-label-map.json', 'r') as f:
    CATEGORY_TO_LABEL = json.load(f)
NUM_CLASSES = len(CATEGORY_TO_LABEL)


with open('sample-distribution.json', 'r') as f:
    _SAMPLE_DISTRIBUTION = json.load(f)


def _shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def train_image_loader(batch_size, normalize=True, augment=False):
    images = []
    categories = []

    # Used only if augment is True
    image_data_generator = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

    with open('input/train.bson', 'rb') as f:
        for d in bson.decode_file_iter(f):
            cid = d['category_id']
            # Convert from category_id to index
            c = CATEGORY_TO_LABEL[cid]
            num_samples = _SAMPLE_DISTRIBUTION[str(cid)]
            # Many categories have less than 1,200 samples.
            # Boost that number to around 1,200 via data augmentation
            iters = 1200 // num_samples if augment else 1
            for pic in d['imgs']:
                img = imread(io.BytesIO(pic['picture']))

                for i in range(iters):
                    if augment:
                        x = image_data_generator.random_transform(img.astype(K.floatx()))
                    else:
                        x = img

                    images.append(x)
                    categories.append(c)

                    if len(categories) >= batch_size:
                        images = np.stack(images)
                        categories = np.stack(categories).squeeze()
                        if normalize:
                           images = (images - 127.5) / 127.5
                        yield images, categories
                        images = []
                        categories = []

    if images:
        images = np.stack(images)
        categories = np.stack(categories).squeeze()
        if normalize:
            images = (images - 127.5) / 127.5
        # print(counter)
        yield images, categories


def feature_loader(paths, batch_size=64):
    """Yields the convolutional features obtained from the training images"""
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
            _shuffle_in_unison(features, categories)
            # Split into mini batches
            num_batches = len(categories) // batch_size
            features = np.array_split(features, num_batches)
            categories = np.array_split(categories, num_batches)
            while categories:
                batch_features = features.pop()
                batch_categories = categories.pop()
                yield batch_features, batch_categories


def test_image_loader(batch_size, normalize=True):
    with open('input/test.bson', 'rb') as f:
        images = []
        ids = []
        for d in bson.decode_file_iter(f):
            for pic in d['imgs']:
                img = imread(io.BytesIO(pic['picture']))
                images.append(img)
                ids.append(d['_id'])

            if len(ids) >= batch_size:
                images = np.stack(images)
                if normalize:
                    images = (images - 127.5) / 127.5
                yield ids, images
                images = []
                ids = []

        if images:
            images = np.stack(images)
            if normalize:
                images = (images - 127.5) / 127.5
            yield ids,
