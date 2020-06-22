import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

input = tf.keras.layers.Input(shape=(32, 32, 3))
# Naive Inception Module
naive_layer01_conv01 = tf.keras.layers.Conv2D(8, (1, 1), padding='SAME', activation='relu')(input)
naive_layer01_conv02 = tf.keras.layers.Conv2D(8, (3, 3), padding='SAME', activation='relu')(input)
naive_layer01_conv03 = tf.keras.layers.Conv2D(8, (5, 5), padding='SAME', activation='relu')(input)
naive_layer01_maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(input)
naive_layer01_concat = tf.keras.layers.Concatenate()([naive_layer01_conv01,
                                                      naive_layer01_conv02,
                                                      naive_layer01_conv03,
                                                      naive_layer01_maxpool]) # 32x32x27

naive_layer02_conv01 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(naive_layer01_concat)
naive_layer02_conv02 = tf.keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu')(naive_layer01_concat)
naive_layer02_conv03 = tf.keras.layers.Conv2D(16, (5, 5), padding='SAME', activation='relu')(naive_layer01_concat)
naive_layer02_maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(naive_layer01_concat)
naive_layer02_concat = tf.keras.layers.Concatenate()([naive_layer02_conv01,
                                                      naive_layer02_conv02,
                                                      naive_layer02_conv03,
                                                      naive_layer02_maxpool]) # 32x32x75

maxpool_layer01 = tf.keras.layers.MaxPool2D((2, 2))(naive_layer02_concat) # 16x16x75

dim_layer01_conv01 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(maxpool_layer01)

dim_layer01_conv02 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(maxpool_layer01)
dim_layer01_conv02 = tf.keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu')(dim_layer01_conv02)

dim_layer01_conv03 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(maxpool_layer01)
dim_layer01_conv03 = tf.keras.layers.Conv2D(16, (5, 5), padding='SAME', activation='relu')(dim_layer01_conv03)

dim_layer01_maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(maxpool_layer01)
dim_layer01_maxpool = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(dim_layer01_maxpool)
dim_layer01_concat = tf.keras.layers.Concatenate()([dim_layer01_conv01,
                                                    dim_layer01_conv02,
                                                    dim_layer01_conv03,
                                                    dim_layer01_maxpool]) #16x16x64

maxpool_layer02 = tf.keras.layers.MaxPool2D((2, 2))(dim_layer01_concat) # 8x8x64
dim_layer02_conv01 = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(maxpool_layer02)

dim_layer02_conv02 = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(maxpool_layer02)
dim_layer02_conv02 = tf.keras.layers.Conv2D(32, (3, 3), padding='SAME', activation='relu')(dim_layer02_conv02)

dim_layer02_conv03 = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(maxpool_layer02)
dim_layer02_conv03 = tf.keras.layers.Conv2D(32, (5, 5), padding='SAME', activation='relu')(dim_layer02_conv03)

dim_layer02_maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(maxpool_layer02)
dim_layer02_maxpool = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(dim_layer02_maxpool)
dim_layer02_concat = tf.keras.layers.Concatenate()([dim_layer02_conv01,
                                                    dim_layer02_conv02,
                                                    dim_layer02_conv03,
                                                    dim_layer02_maxpool]) #8x8x128

global_avgpool = tf.keras.layers.GlobalAveragePooling2D()(dim_layer02_concat)

# flatten = tf.keras.layers.Flatten()(dim_layer02_concat)
output = tf.keras.layers.Dense(10, activation='softmax')(global_avgpool)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=10,
          validation_data=(x_test, y_test))
