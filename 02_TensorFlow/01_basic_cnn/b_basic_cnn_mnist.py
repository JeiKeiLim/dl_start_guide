import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tfhelper.tensorboard import get_tf_callbacks, run_tensorboard, wait_ctrl_c


############# Data Preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# plt.imshow(x_test[0], cmap='gray')
# plt.show()

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
############# Data Preparation

######### Model build
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='VALID', activation='relu'),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='VALID', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='VALID', activation='relu'),
    # tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

callbacks, log_root = get_tf_callbacks("./export/tflog", tboard_callback=True,
                                       confuse_callback=True, x_test=x_test, y_test=np.argmax(y_test, axis=1),
                                       modelsaver_callback=False)
run_tensorboard(log_root)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# input = tf.keras.layers.Input(shape=(784,))
# hidden_layer01 = tf.keras.layers.Dense(256, activation='relu')(input)
# hidden_layer02 = tf.keras.layers.Dense(128, activation='relu')(hidden_layer01)
# output = tf.keras.layers.Dense(10, activation='softmax')(hidden_layer02)
#
# model = tf.keras.models.Model(inputs=input, outputs=output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
######### Model build

model.summary()

#### Training
model.fit(x_train, y_train, batch_size=256, epochs=50,
          validation_data=(x_test, y_test),
          callbacks=callbacks)
#### Training

test_prediction = model.predict(x_test)

wait_ctrl_c()
