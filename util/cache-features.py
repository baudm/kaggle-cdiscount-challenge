#!/usr/bin/env python

import uuid

import numpy as np

from keras.applications import Xception

from buffering import buffered_gen_threaded as buffered
import data


def main():
    model = Xception(include_top=False, input_shape=(180, 180, 3), pooling='avg')

    batch_size = 256
    loader = buffered(data.train_image_loader(batch_size), 10)

    features = []
    cats = []

    def save(features, cats):
        name = str(uuid.uuid4()) + '.npz'
        features = np.array(features)
        cats = np.array(cats).flatten()
        print('save:', name)
        with open('features/' + name, 'wb') as f:
            np.savez(f, features=features, categories=cats)

    for img, cat in loader:
        f = model.predict_on_batch(img)
        features.append(f)
        cats.append(cat)

        if len(cats) >= 256:
            save(features, cats)
            features = []
            cats = []

    save(features, cats)


if __name__ == '__main__':
    main()
