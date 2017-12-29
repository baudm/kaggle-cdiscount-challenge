#!/usr/bin/env python

import json
import bson


def loader():
    stats = {}
    counter = 0
    with open('input/train.bson', 'rb') as f:
        for d in bson.decode_file_iter(f):
            c = d['category_id']
            count = stats.get(c, 0)
            count += len(d['imgs'])
            stats[c] = count
            counter += len(d['imgs'])

    return stats


def main():
    stats = loader()
    with open('sample-distribution.json', 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    main()
