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
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='SAME', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
######### Model build

model.summary()

#### Training
model.fit(x_train, y_train, batch_size=64, epochs=5,
          validation_data=(x_test, y_test))
#### Training


def get_cam_image(model_, x, img_size=(28, 28), layer_idx=None):
    if layer_idx is None:
        for layer_idx in range(len(model.layers) - 1, -1, -1):
            if type(model.layers[layer_idx]) == tf.keras.layers.Conv2D:
                break

    cam_model_ = tf.keras.models.Model(model_.inputs, [model_.layers[layer_idx].output, model_.output])
    conv_out, model_out = cam_model_(x)

    cam_images_ = np.zeros((x.shape[0], img_size[0], img_size[1]))

    for i, outs in enumerate(zip(conv_out, model_out)):
        c_out, m_out = outs
        predict_idx = np.argmax(m_out)
        chosen_weight = model_.layers[-1].weights[0][:, predict_idx]

        cam_img_ = np.zeros(c_out.shape[0:2])

        for j in range(c_out.shape[2]):
            cam_img_ += c_out[:, :, j] * chosen_weight[j]

        cam_images_[i] = cv2.resize(cam_img_.numpy(), img_size)

    return cam_images_


test_index = np.arange(10)

cam_img = get_cam_image(model, x_test[test_index], img_size=(28, 28))

for i, idx in enumerate(test_index):
    plt.subplot(1, 2, 1)
    plt.imshow(1-x_test[idx], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(1-x_test[idx], cmap='gray')
    plt.imshow(cam_img[i], cmap='jet', alpha=0.5)
    plt.show()

