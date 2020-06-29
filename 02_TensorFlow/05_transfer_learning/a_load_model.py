import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("Custom Image Path")
image = cv2.resize(image, (224, 224))
image = np.expand_dims(image, axis=0)
image = image / 255

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')

model.summary()

result = model.predict(image)
result_sorted = np.argsort(result, axis=1)[0][::-1]

print(result_sorted[:5])
