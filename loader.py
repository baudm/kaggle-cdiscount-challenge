
def loader(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            features = archive['features']
            categories = archive['categories']
            del archive
            #features = features.reshape(256*256, -1)
            #categories = categories.flatten()
            # Split into mini batches
            num_batches = int(len(categories) / batch_size)
            #half = int(np.ceil(num_batches/2.))
            features = np.array_split(features, num_batches)
            categories = np.array_split(categories, num_batches)
            shuffle_in_unison_scary(features, categories)
            #can_preload = True
            while categories:
                batch_features = features.pop()
                batch_categories = categories.pop()
                # convert to one-hot representation
                #batch_categories = np.stack([to_categorical(c, num_classes) for c in batch_categories]).squeeze()
                yield batch_features, batch_categories
                # if can_preload and len(features) <= half and i + 1 < len(files):
                #     can_preload = False
                #     # preload next file
                #     np.load(files[i+1])['features']