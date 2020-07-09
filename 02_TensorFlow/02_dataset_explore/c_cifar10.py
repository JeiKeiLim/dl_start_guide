import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

n_image = 10

plt.figure(figsize=(30, 30))
for i in range(10):
    class_indices = np.argwhere(y_train.flatten() == i).flatten()

    for j in range(n_image):
        plt.subplot(10, n_image, i*n_image+j+1)

        random_index = np.random.randint(0, class_indices.shape[0])
        random_index_in_class = class_indices[random_index]

        plt.imshow(x_train[random_index_in_class])
        plt.axis('off')
        # plt.title("Class Number : {}, Data Index : {}".format(i, random_index_in_class))
plt.show()

