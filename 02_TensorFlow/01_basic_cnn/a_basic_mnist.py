import tensorflow as tf
import numpy as np

############# Data Preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
############# Data Preparation


######### Model build
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

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
model.fit(x_train, y_train, batch_size=64, epochs=10,
          validation_data=(x_test, y_test))
#### Training

test_prediction = model.predict(x_test)

truth = np.argmax(y_test, axis=1)
prediction = np.argmax(test_prediction, axis=1)

print("Prediction Accuracy: {:.2f}%".format((np.sum(truth == prediction) / y_test.shape[0]) * 100))
