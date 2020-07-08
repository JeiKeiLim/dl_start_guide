import tensorflow as tf

############# Data Preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
############# Data Preparation

######### Model build
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)), # 28x28
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='VALID', activation='relu'), #26x26
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='VALID', activation='relu'), #24x24
    tf.keras.layers.MaxPool2D((2, 2)), #12x12
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='VALID', activation='relu'), #10x10
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

######### Model build

model.summary()

#### Training
model.fit(x_train, y_train, batch_size=256, epochs=10,
          validation_data=(x_test, y_test),)
#### Training

test_prediction = model.predict(x_test)

