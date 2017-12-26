#!/usr/bin/env python

import bson
#from skimage.data import imread
import cv2
import pickle

import io
import uuid
import os.path
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Conv2D, InputLayer, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten, Activation, Input
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import random

def pack(outdir, images, categories):
    name = str(uuid.uuid4())
    pack = os.path.join(outdir, name + '.npz')
    images = np.stack(images)
    categories = np.stack(categories)
    with open(pack, 'wb') as f:
        np.savez(f, images=images, categories=categories)
    print('bundled:', name)


with open('category-table', 'rb') as f:
    lookup_table = pickle.load(f)
num_classes = len(lookup_table)


VAL_BATCH_SIZE = 128
VAL_SET = 51200
VAL_STEPS = int(VAL_SET/VAL_BATCH_SIZE)

def train_loader(batch_size, normalize=True):
    while True:
        images = []
        categories = []
        # make first N images the validation set
        skip = VAL_SET
        with open('input/train.bson', 'rb') as f:
            data = bson.decode_file_iter(f)
            for d in data:
                c = d['category_id']
                # Convert from category_id to index
                c = lookup_table[c]
                c = to_categorical(c, num_classes)
                ### Accelerate by using just 1 image per product
                for pic in d['imgs']:
                    if skip:
                        skip -= 1
                        continue
                    # Randomly skip 3/4 of training samples
                    #if random.randint(0, 3) != 0:
                    #    continue

                    img = cv2.imdecode(np.fromstring(pic['picture'], 'uint8'), cv2.IMREAD_COLOR)

                    images.append(img)
                    categories.append(c)

                    if len(categories) >= batch_size:
                        images = np.stack(images)
                        categories = np.stack(categories).squeeze()
                        if normalize:
                            images = (images - 127.5) / 127.5
                        yield images, categories
                        images = []
                        categories = []


import cv2

def val_loader(batch_size, normalize=True):
    while True:
        images = []
        categories = []
        counter = 0
        with open('input/splitted.1.bson', 'rb') as f:
            data = bson.decode_file_iter(f)
            for d in data:
                c = d['category_id']
                # Convert from category_id to index
                c = lookup_table[c]
                c = to_categorical(c, num_classes)
                ### Accelerate by using just 1 image per product
                for pic in d['imgs']:
                    # Randomly skip 3/4 of training samples
                    #if random.randint(0, 3) != 0:
                    #    continue

                    img = cv2.imdecode(np.fromstring(pic['picture'], 'uint8'), cv2.IMREAD_COLOR)

                    images.append(img)
                    categories.append(c)

                    if len(categories) >= batch_size:
                        images = np.stack(images)
                        categories = np.stack(categories).squeeze()
                        if normalize:
                            images = (images - 127.5) / 127.5
                        yield images, categories
                        images = []
                        categories = []

                    counter += 1
                    if counter >= VAL_SET:
                        break

                if counter >= VAL_SET:
                    break



def model1():
    model = Sequential()
    model.add(InputLayer(input_shape=(180, 180, 3)))

    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(256, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(512, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, 3, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(1024, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(1024, 2, padding='valid', activation='selu', kernel_initializer='lecun_normal'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(2048, 1, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(4096, 1, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    # model.add(MaxPooling2D())

    # model.add(BatchNormalization())
    # model.add(Conv2D(8092, 1, padding='same', activation='selu', kernel_initializer='lecun_normal'))

    # model.add(MaxPooling2D())


    model.add(Flatten())

    model.add(Dropout(0.2))

    # model.add(Dense(4096, activation='selu'))

    # model.add(Dropout(0.2))

    # model.add(Dense(2*4096, activation='selu'))

    model.add(Dense(num_classes, activation='softmax', kernel_initializer='lecun_normal'))

    return model


def model_resnet():
    total_layers = 25  # Specify how deep we want our network
    units_between_stride = int(total_layers / 5)

    def resUnit(input_layer, i):
        y = BatchNormalization()(input_layer)
        y = Activation('selu')(y)
        y = Conv2D(64, 3, padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('selu')(y)
        y = Conv2D(64, 3, padding='same')(y)
        output = input_layer + y
        return output

    input_layer = Input(shape=(180, 180, 3))

    layer1 = Conv2D(64, 3)(input_layer)

    for i in range(5):
        for j in range(units_between_stride):
            layer1 = resUnit(layer1, j + (i * units_between_stride))
        layer1 = Conv2D(64, 3, strides=(2, 2))(layer1)

    top = Conv2D(num_classes, 3, activation='softmax')(layer1)
    top = Flatten()(top)
    model = Model(input_layer, top)
    return model


import resnet
from keras.optimizers import SGD

from keras.layers import GlobalAveragePooling2D, AveragePooling2D

from inception_resnet_v2 import InceptionResNetV2, conv2d_bn

from keras.callbacks import TensorBoard


def modify_inception_reset_v2(x):

    return x

def make_InceptionResNetV2(input_shape):
    base_model = InceptionResNetV2(include_top=False, input_shape=input_shape, pooling='avg')
    x = base_model.output
    # x = Conv2D(3072, 3)(x)
    # x = AveragePooling2D()(x)
    x = Dropout(0.2)(x)
    # x = conv2d_bn(x, 3072, 1, name='conv_final')
    # x = GlobalAveragePooling2D()(x)
    x = Dense(8192, activation='relu')(x)
    x = Dropout(0.2)(x)
    return base_model, x



from keras.applications import Xception
from keras.layers import SeparableConv2D, GlobalAveragePooling2D

def make_Xception(input_shape):
    base_model = Xception(include_top=False, input_shape=input_shape, pooling='avg')
    x = base_model.output

    # x = Dropout(0.2)(x)
    #x = SeparableConv2D(4096, (3, 3), use_bias=False, name='block15_sepconv')(x)
    #x = BatchNormalization(name='block15_sepconv_bn')(x)
    #x = Activation('relu', name='block15_sepconv_act')(x)

    # x = SeparableConv2D(4096*2, (3, 3), use_bias=False, name='block16_sepconv')(x)
    # x = BatchNormalization(name='block16_sepconv_bn')(x)
    # x = Activation('relu', name='block16_sepconv_act')(x)

    #x = GlobalAveragePooling2D()(x)
    return base_model, x


def make_model(input_shape):
    base_model, x = make_Xception(input_shape)
    classifier = load_model('model.52.hdf5', compile=False)
    #x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers:
        if layer.name.startswith('block14'):
            layer.trainable = True

    model = Model(base_model.input, classifier(x))

    return model


from keras.models import load_model
import sys


def main():
    input_shape = (180, 180, 3)

    if len(sys.argv) == 2:
        model = load_model(sys.argv[1], compile=False)
    else:
        model = make_model(input_shape)



    model.summary()

    #print('dropout:', model.layers[-2].rate)

    checkpoint = ModelCheckpoint('/home/darwin/Projects/kaggle/cdiscount/model.{epoch:02d}.hdf5')
    #lr_scheduler = ReduceLROnPlateau(patience=2)

    adam = Adam(0.01)
    sgd = SGD(0.05, 0.9, nesterov=True)


    # rmsprop @50% ETA: 947s - loss: 6.4258 - acc: 0.3571^
    # 119/1007 [==>...........................] - ETA: 1676s - loss: 3.5423 - acc: 0.4910^
    # 192/1007 [====>.........................] - ETA: 1538s - loss: 3.1040 - acc: 0.4882
    model.compile(sgd, 'categorical_crossentropy', metrics=['accuracy'])


    num_samples = 12371293 # skipping 75% of the samples
    batch_size = 512
    steps_per_epoch = 1000
    epochs = int(np.round(num_samples / (batch_size * steps_per_epoch)))

    #tensorboard = TensorBoard(log_dir='/tmp/logs', batch_size=batch_size)

    model.fit_generator(train_loader(batch_size), steps_per_epoch, epochs, callbacks=[checkpoint]
                        ,validation_data=val_loader(VAL_BATCH_SIZE), validation_steps=20)
    #loader()


if __name__ == '__main__':
    main()
