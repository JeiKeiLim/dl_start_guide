import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
# y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)


input = tf.keras.layers.Input(shape=(32, 32, 3))
# reshape = tf.keras.layers.Reshape((28, 28, 1))(input)


def vgg_block(input, n_conv, n_filter, filter_size=(3, 3), reduce_size=True):
    layer = input
    for i in range(n_conv):
        layer = tf.keras.layers.Conv2D(n_filter, filter_size, padding='SAME', activation='relu')(layer)

    if reduce_size:
        layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
    return layer


vgg_block01 = vgg_block(input, 2, 16)
vgg_block02 = vgg_block(vgg_block01, 2, 32)
vgg_block03 = vgg_block(vgg_block02, 3, 64)

flatten = tf.keras.layers.Flatten()(vgg_block03)
dense01 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
dense02 = tf.keras.layers.Dense(1024, activation='relu')(dense01)
output = tf.keras.layers.Dense(10, activation='softmax')(dense02)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10,
          validation_data=(x_test, y_test))

