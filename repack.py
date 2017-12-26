#!/usr/bin/env python

import numpy as np
import glob



def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

import uuid

def save(b_f, b_c):
    name = str(uuid.uuid4()) + '.npz'
    print('save', name)
    with open('packed/' + name, 'wb') as g:
        np.savez(g, features=np.stack(b_f), categories=np.stack(b_c))
    b_f.clear()
    b_c.clear()


def main():
    files = glob.glob('/var/local/features/**/*.npz', recursive=True)
    np.random.shuffle(files)
    f_buckets = ([], [], [], [], [], [], [], [])
    c_buckets = ([], [], [], [], [], [], [], [])
    max_bucket_size = 256*256
    for npz in files:
        print('processing',npz)
        ar = np.load(npz)
        f = ar['features']
        c = ar['categories']
        shuffle_in_unison_scary(f, c)
        for i in range(len(c)):
            j = np.random.randint(len(f_buckets))
            b_f = f_buckets[j]
            b_c = c_buckets[j]
            b_f.append(f[i])
            b_c.append(c[i])
            if len(b_c) >= max_bucket_size:
                save(b_f, b_c)

    all_f = []
    all_c = []
    for i in range(len(f_buckets)):
        all_f.extend(f_buckets[i])
        all_c.extend(c_buckets[i])

    del f_buckets, c_buckets


    b_f = []
    b_c = []
    while all_f:
        f = all_f.pop()
        c = all_c.pop()
        b_f.append(f)
        b_c.append(c)
        if len(b_c) >= max_bucket_size:
            save(b_f, b_c)

    if b_f:
        save(b_f, b_c)


if __name__ == '__main__':
    main()