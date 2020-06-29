import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

model = tf.keras.applications.MobileNet(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
model.trainable = False

t_model = tf.keras.models.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

t_model.summary()

t_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='sparse_categorical_crossentropy',
                metrics='accuracy')
# Transfer Learning
t_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Fine-Tuning
t_model.trainable = True
t_model.summary()
t_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics='accuracy')
t_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

