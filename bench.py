#!/usr/bin/env python

import bson

f= open('input/train.bson', 'rb')
data = bson.decode_file_iter(f)
c = 0
for d in data:
    c += len(d['imgs'])
    if c >= 1000000:
        break
f.close()
