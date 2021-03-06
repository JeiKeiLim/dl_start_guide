import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tfhelper.tensorboard import get_tf_callbacks, run_tensorboard, wait_ctrl_c


############# Data Preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

mean_train = x_train.mean()
std_train = x_train.std()

x_train = x_train - mean_train
x_train = x_train / std_train

# x_train, x_test = x_train / 255.0, x_test / 255.0

# plt.imshow(x_test[0], cmap='gray')
# plt.show()

# y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
############# Data Preparation

lr = 0.25
print("Learning rate: {}".format(lr))
######### Model build
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal()),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal()),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal()),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal()),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal()),
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal()),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.RandomNormal())
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

######### Model build

# model.summary()

#### Training
# model.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_test, y_test))
model.fit(x_train, y_train, batch_size=256, epochs=10, validation_split=0.2)
#### Training

model.evaluate(x_test, y_test)
model.evaluate((x_test-mean_train)/std_train, y_test)

# test_prediction = model.predict(x_test)


