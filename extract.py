#!/usr/bin/env python
import pickle
import bson
import cv2

import numpy as np

with open('category-table', 'rb') as f:
    lookup_table = pickle.load(f)
num_classes = len(lookup_table)


VAL_BATCH_SIZE = 256
VAL_SET = 242*512
VAL_STEPS = int(VAL_SET/VAL_BATCH_SIZE)


from keras.applications.xception import preprocess_input
from keras.applications import Xception

def val_loader(batch_size, normalize=True):
    images = []
    categories = []
    counter = 0
    with open('input/train.bson', 'rb') as f:
        data = bson.decode_file_iter(f)
        for d in data:
            c = d['category_id']
            # Convert from category_id to index
            c = lookup_table[c]

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
                    print(counter)
                    yield images, categories
                    images = []
                    categories = []

                counter += 1
                if counter >= VAL_SET:
                    break

            if counter >= VAL_SET:
                break

from skimage.data import imread
import io

def train_loader(batch_size, normalize=True):
    images = []
    categories = []
    # make first N images the validation set
    skip = VAL_SET
    counter = 0
    with open('input/train.bson', 'rb') as f:
        data = bson.decode_file_iter(f)
        for d in data:
            c = d['category_id']
            # Convert from category_id to index
            c = lookup_table[c]
            #c = to_categorical(c, num_classes)
            ### Accelerate by using just 1 image per product
            for pic in d['imgs']:
                if skip:
                    skip -= 1
                    continue
                # Randomly skip 3/4 of training samples
                #if random.randint(0, 3) != 0:
                #    continue

                img = imread(io.BytesIO(pic['picture']))

                images.append(img)
                categories.append(c)

                counter += 1

                if len(categories) >= batch_size:
                    images = np.stack(images)
                    categories = np.stack(categories).squeeze()
                    if normalize:
                       images = (images - 127.5) / 127.5
                    print(counter)
                    yield images, categories
                    images = []
                    categories = []


def main():
    model = Xception(include_top=False, input_shape=(180, 180, 3), pooling='avg')
    a = 12371293-VAL_STEPS
    steps = int(a/VAL_BATCH_SIZE)
    loader = val_loader(VAL_BATCH_SIZE)
    features = []
    cats = []
    i = 0

    for img, cat in loader:
        f = model.predict_on_batch(img)
        features.append(f)
        cats.append(cat)

    #if len(cats) >= 256:
    n = '{:02d}'.format(i)
    features = np.array(features).reshape(len(cats) * 256, -1)
    cats = np.array(cats).flatten()
    with open('val-features.' +n+'.npz', 'wb') as f:
        np.savez(f, features=features, categories=cats)
    features = []
    cats = []
    i += 1

    #features = np.stack(features)
    #cats = np.stack(cats)



if __name__ == '__main__':
    main()