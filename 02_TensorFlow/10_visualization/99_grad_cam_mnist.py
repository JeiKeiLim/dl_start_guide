import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

############# Data Preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
plt.imshow(x_test[0], cmap='gray')
plt.show()

y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
############# Data Preparation

######### Model build
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
######### Model build

model.summary()

#### Training
model.fit(x_train, y_train, batch_size=64, epochs=10,
          validation_data=(x_test, y_test))
#### Training

# test_prediction = model.predict(x_test)
cam_model = tf.keras.models.Model(model.inputs, [model.layers[1].output, model.output])
test_class_idx = 1

with tf.GradientTape() as tape:
    out_c, out_p = cam_model(x_test)
    loss = out_p[:, test_class_idx]

grads = tape.gradient(loss, out_c)[0]

idx = np.where(np.argmax(y_test, axis=1) == test_class_idx)
idx = idx[0][np.random.randint(0, idx[0].shape[0])]

output = out_c[idx]

gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

weights = tf.reduce_mean(guided_grads, axis=(0, 1))

cam = np.ones(output.shape[0: 2], dtype = np.float32)

for i, w in enumerate(weights):
    cam += w * output[:, :, i]

cam = cv2.resize(cam.numpy(), (28, 28))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
output_image = cv2.addWeighted(cv2.cvtColor((x_test[idx]*255).astype('uint8'), cv2.COLOR_GRAY2BGR), 0.5, cam, 0.5, 0)

plt.imshow(output_image)
plt.show()
