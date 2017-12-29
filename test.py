#!/usr/bin/env python

import sys

import csv
from threading import Thread
from queue import Queue

import numpy as np

from keras.models import load_model

import data
import models

# Inverse mapping
LABEL_TO_CATEGORY = dict(zip(data.CATEGORY_TO_LABEL.values(), data.CATEGORY_TO_LABEL.keys()))


def main():
    top_model = load_model(sys.argv[1]) if len(sys.argv) == 2 else None
    model = models.create_full_model(top_model)

    q = Queue()
    iqueue = Queue(3)

    def dump():
        with open('predictions.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['_id', 'category_id'])
            while True:
                d = q.get()
                if d is None:
                    break
                i, p = d

                cats = np.argmax(p, axis=-1)
                cats = list(map(LABEL_TO_CATEGORY.get, cats))

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


    batch_size = 256

    def process():
        while True:
            d = iqueue.get()
            if d is None:
                break
            i, img = d
            p = model.predict_on_batch(img)
            q.put((i, p))
        q.put(None)

    pt = Thread(target=process)
    pt.start()

    for i, img in data.test_image_loader(batch_size):
        iqueue.put((i, img))

    iqueue.put(None)
    iqueue.join()
    pt.join()
    q.join()
    t.join()


if __name__ == '__main__':
    main()
