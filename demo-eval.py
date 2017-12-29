#!/usr/bin/env python

from models import create_full_model
import data


def main():
    model = create_full_model()
    model.summary()

    model.compile('sgd', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    batch_size = 10
    num_samples = 110
    loader = data.train_image_loader(batch_size, demo=True)
    result = model.evaluate_generator(loader, num_samples // batch_size)
    print(model.metrics_names, result)


if __name__ == '__main__':
    main()
