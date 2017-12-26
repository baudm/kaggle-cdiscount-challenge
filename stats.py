#!/usr/bin/env python

import pickle
import bson

with open('category-table', 'rb') as f:
    lookup_table = pickle.load(f)
num_classes = len(lookup_table)



def loader():
    stats = {}
    counter = 0
    with open('input/train.bson', 'rb') as f:
        data = bson.decode_file_iter(f)
        for d in data:
            c = d['category_id']
            count = stats.get(c, 0)
            count += len(d['imgs'])
            stats[c] = count
            counter += len(d['imgs'])
            if counter % 123712 == 0:
                print(counter)

    return stats

import json

def main():
    stats = loader()
    with open('stats.json', 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    main()