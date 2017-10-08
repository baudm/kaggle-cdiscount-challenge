#!/usr/bin/env python

import bson
import sys
from keras.models import load_model
import pickle

with open('category-table', 'rb') as f:
    lookup_table = pickle.load(f)

# swap keys and values
lookup_table = dict(zip(lookup_table.values(),lookup_table.keys()))

num_classes = len(lookup_table)

import io
from skimage.data import imread
import numpy as np
from keras.utils import GeneratorEnqueuer

def loader(batch_size, normalize=True):
    images = []
    ids = []
    with open('input/test.bson', 'rb') as f:
        data = bson.decode_file_iter(f)
        for d in data:
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


import csv
from threading import Thread
from queue import Queue

def main():
    if len(sys.argv) != 2:
        print('Model required')
        exit(1)

    q = Queue()

    def dump():
        with open('out.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['_id', 'category_id'])
            while True:
                d = q.get()
                if d is None:
                    break
                i, p = d

                cats = np.argmax(p, axis=-1)
                cats = list(map(lookup_table.get, cats))

                a = {}
                for j in range(len(i)):
                    l = a.setdefault(i[j], [])
                    l.append(cats[j])

                for prod_id, cat_ids in a.items():
                    cat_ids, counts = np.unique(cat_ids, return_counts=True)
                    best_idx = np.argmax(counts)
                    best_cat = cat_ids[best_idx]
                    writer.writerow([prod_id, best_cat])



    t = Thread(target=dump)
    t.start()

    model = load_model(sys.argv[1])

    batch_size = 512

    for i, img in loader(batch_size):
        p = model.predict(img, len(i))
        q.put((i, p))

    q.put(None)
    q.join()
    t.join()


if __name__ == '__main__':
    main()
