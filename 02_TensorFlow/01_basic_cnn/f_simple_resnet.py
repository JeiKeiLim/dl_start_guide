import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

input = tf.keras.layers.Input(shape=(32, 32, 3))


def resnet_block(input, n_filter, reduce_size=False, filter_size=(3, 3), dropout_rate=0.5):
    layer = input

    stride_size = (2, 2) if reduce_size else (1, 1)

    layer = tf.keras.layers.Conv2D(n_filter, filter_size, padding='SAME', strides=stride_size, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    layer = tf.keras.layers.Conv2D(n_filter, filter_size, padding='SAME', activation='relu')(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    input_2 = tf.keras.layers.Conv2D(n_filter, (1, 1), padding='SAME', activation='relu')(input)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    layer = tf.keras.layers.BatchNormalization()(layer)

    if reduce_size:
        input_2 = tf.keras.layers.MaxPool2D((2, 2))(input_2)

    layer = tf.keras.layers.Add()([input_2, layer])

    return layer


def resnet_bottle_neck_block(input, n_filter, filter_size=(3, 3), dropout_rate=0.5):
    layer = input

    layer = tf.keras.layers.Conv2D(n_filter, (1, 1), padding='SAME', activation='relu')(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    layer = tf.keras.layers.Conv2D(n_filter, filter_size, padding='SAME', activation='relu')(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    # batch size x width x height x # of channel
    layer = tf.keras.layers.Conv2D(input.shape[3], (1, 1), padding='SAME', activation='relu')(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.Add()([layer, input])

    return layer


resnet_block01_01 = resnet_bottle_neck_block(input, 16)
resnet_block01_02 = resnet_bottle_neck_block(resnet_block01_01, 16)
resnet_block01_03 = resnet_bottle_neck_block(resnet_block01_02, 16)
resnet_block02_01 = resnet_block(resnet_block01_03, 32, reduce_size=True)
resnet_block02_02 = resnet_bottle_neck_block(resnet_block02_01, 32)
resnet_block02_03 = resnet_bottle_neck_block(resnet_block02_02, 32)
resnet_block03_01 = resnet_block(resnet_block02_03, 64, reduce_size=True)
resnet_block03_02 = resnet_bottle_neck_block(resnet_block03_01, 64)
resnet_block03_03 = resnet_bottle_neck_block(resnet_block03_02, 64)

global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(resnet_block03_03)
output = tf.keras.layers.Dense(10, activation='softmax')(global_avg_pooling)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=50,
          validation_data=(x_test, y_test))
