import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt


# Random Noise Function
def add_random_noise(x):
    x = x + np.random.normal(size=x.shape) * np.random.uniform(1, 5)
    x = x - x.min()
    x = x / x.max()

    return x * 255.0


# Set your dataset directory
TRAINING_DIR = "/Users/jeikei/Documents/deep_summer_dataset/webcam_dataset"
VALIDATION_DIR = "/Users/jeikei/Documents/deep_summer_dataset/test-set"

batch_size = 8

# Image Data Generator with Augmentation
training_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      brightness_range=(0.5, 1.3),
      horizontal_flip=True,
      fill_mode='nearest',
      preprocessing_function=add_random_noise)

validation_datagen = ImageDataGenerator(rescale = 1./255)

# Reading images from directory and pass them to the model
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=batch_size,
    target_size=(224,224),
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=batch_size,
    target_size=(224,224),
    class_mode='categorical'
)

# Plotting the augmented images
img, label = next(train_generator)
plt.figure(figsize=(20, 20))

for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(img[i])
    plt.title(label[i])
    plt.axis('off')

plt.show()

# Load pre-trained base model.
base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# Unfreeze the base model
base_model.trainable = True

# Freeze the base model except the last layer of unfreeze_n_layer
unfreeze_n_layer = 3
for i in range(0, len(base_model.layers) - unfreeze_n_layer):
    base_model.layers[i].trainable = False

# Add Custom layers
out_layer = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', activation=None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer)

out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)

out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)

# Make New Model
model = tf.keras.models.Model(base_model.input, out_layer)

# Load saved weights
model.load_weights("saved_model.h5")

model.summary()

# Set lower learning rate for fine tuning
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Fine tuning
history = model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose=1)

# Save the fine-tuned model
model.save("saved_model_finetune.h5")

