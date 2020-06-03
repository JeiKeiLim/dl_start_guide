import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.imshow(x_test[0], cmap='gray')
plt.show()

conv_img = x_test[0]
conv_filter = np.array([[-1, 0, 1],
                        [-1, 1, 1],
                        [-1, 0, 1]])

conv_filter = conv_filter / conv_filter.sum()

new_conv_img = np.zeros((conv_img.shape[0]-conv_filter.shape[0], conv_img.shape[1]-conv_filter.shape[1]))

for i in range(new_conv_img.shape[0]):
    y1, y2 = i, i + conv_filter.shape[0]

    for j in range(new_conv_img.shape[1]):
        x1, x2 = j, j + conv_filter.shape[1]

        new_conv_img[i, j] = np.dot(conv_img[y1:y2, x1:x2].flatten(), conv_filter.flatten())

plt.imshow(new_conv_img, cmap='gray')
plt.show()
