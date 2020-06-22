import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

input = tf.keras.layers.Input(shape=(32, 32, 3)) # 32x32x3

layer01_conv01 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', strides=(1, 1), activation='relu')(input) # 32x32x16
layer01_conv02 = tf.keras.layers.Conv2D(16, (3, 3), padding='SAME', strides=(1, 1), activation='relu')(layer01_conv01) # 32x32x16
layer01_conv03 = tf.keras.layers.Conv2D(3, (1, 1), padding='SAME', strides=(1, 1), activation='relu')(layer01_conv02) # 32x32x3
layer01_residual = tf.keras.layers.Add()([input, layer01_conv03]) # 32x32x3
layer01_batchnorm = tf.keras.layers.BatchNormalization()(layer01_residual)

layer02_conv01 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', strides=(1, 1), activation='relu')(layer01_batchnorm) #32x32x16
layer02_conv02 = tf.keras.layers.Conv2D(32, (3, 3), padding='SAME', strides=(1, 1), activation='relu')(layer02_conv01) #32x32x32
layer02_conv03 = tf.keras.layers.Conv2D(32, (3, 3), padding='SAME', strides=(1, 1), activation='relu')(layer02_conv02) #32x32x32
layer02_conv_input = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', strides=(1, 1), activation='relu')(layer01_batchnorm) #32x32x32
layer02_residual = tf.keras.layers.Add()([layer02_conv_input, layer02_conv03])
layer02_batchnorm = tf.keras.layers.BatchNormalization()(layer02_residual)

bt_layer01_conv01 = tf.keras.layers.Conv2D(64, (1, 1), padding='SAME', activation='relu')(layer02_batchnorm)
bt_layer01_conv02 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')(bt_layer01_conv01)
bt_layer01_conv03 = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(bt_layer01_conv02)
bt_layer01_residual = tf.keras.layers.Add()([layer02_batchnorm, bt_layer01_conv03])
bt_layer01_batchnorm = tf.keras.layers.BatchNormalization()(bt_layer01_residual)

bt_layer02_conv01 = tf.keras.layers.Conv2D(64, (1, 1), padding='SAME', activation='relu')(bt_layer01_batchnorm)
bt_layer02_conv02 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')(bt_layer02_conv01)
bt_layer02_conv03 = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(bt_layer02_conv02)
bt_layer02_residual = tf.keras.layers.Add()([bt_layer01_residual, bt_layer02_conv03])
bt_layer02_batchnorm = tf.keras.layers.BatchNormalization()(bt_layer02_residual)

layer03_conv01 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=(2, 2), activation='relu')(bt_layer02_batchnorm)
layer03_conv02 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=(1, 1), activation='relu')(layer03_conv01)
layer03_conv03 = tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=(1, 1), activation='relu')(layer03_conv02)
layer03_maxpool_input = tf.keras.layers.MaxPool2D((2, 2))(bt_layer02_residual)
layer03_conv_input = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', strides=(1, 1), activation='relu')(layer03_maxpool_input)
layer03_residual = tf.keras.layers.Add()([layer03_conv_input, layer03_conv03])
layer03_batchnorm = tf.keras.layers.BatchNormalization()(layer03_residual)

bt_layer03_conv01 = tf.keras.layers.Conv2D(64, (1, 1), padding='SAME', activation='relu')(layer03_batchnorm)
bt_layer03_conv02 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')(bt_layer03_conv01)
bt_layer03_conv03 = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', activation='relu')(bt_layer03_conv02)
bt_layer03_residual = tf.keras.layers.Add()([layer03_residual, bt_layer03_conv03])
bt_layer03_batchnorm = tf.keras.layers.BatchNormalization()(bt_layer03_residual)

global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(bt_layer03_batchnorm)
output = tf.keras.layers.Dense(10, activation='softmax')(global_avg_pooling)

model = tf.keras.models.Model(input, output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10,
          validation_data=(x_test, y_test))
