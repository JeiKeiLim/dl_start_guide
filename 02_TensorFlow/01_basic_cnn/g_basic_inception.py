import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

input_layer = tf.keras.layers.Input(shape=(32, 32, 3))

stem_layer = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='SAME', activation='relu')(input_layer)
# Max Pooling Here on Original Inception (3x3, stride=2)
stem_layer = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='SAME', activation='relu')(stem_layer)
# Max Pooling Here on Original Inception (3x3, stride=2)

inception_block3a_1x1 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='SAME', activation='relu')(stem_layer)

inception_block3a_3x3 = tf.keras.layers.Conv2D(48, (1, 1), padding='SAME', activation='relu')(stem_layer)
inception_block3a_3x3 = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu')(inception_block3a_3x3)

inception_block3a_5x5 = tf.keras.layers.Conv2D(8, (1, 1), padding='SAME', activation='relu')(stem_layer)
inception_block3a_5x5 = tf.keras.layers.Conv2D(16, (5, 5), padding='SAME', activation='relu')(inception_block3a_5x5)

inception_block3a_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(stem_layer)
inception_block3a_pool = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(inception_block3a_pool)

inception_block3a = tf.keras.layers.Concatenate()([inception_block3a_1x1, inception_block3a_3x3,
                                                   inception_block3a_5x5, inception_block3a_pool]) # 32x32x128
# inception_block3b Here on Original Inception
max_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='SAME')(inception_block3a) # 16x16x128

inception_block4a_1x1 = tf.keras.layers.Conv2D(64, (1, 1), padding='SAME', activation='relu')(max_pool)

inception_block4a_3x3 = tf.keras.layers.Conv2D(64, (1, 1), padding='SAME', activation='relu')(max_pool)
inception_block4a_3x3 = tf.keras.layers.Conv2D(96, (3, 3), padding='SAME', activation='relu')(inception_block4a_3x3)

inception_block4a_5x5 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(max_pool)
inception_block4a_5x5 = tf.keras.layers.Conv2D(48, (5, 5), padding='SAME', activation='relu')(inception_block4a_5x5)

inception_block4a_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(max_pool)
inception_block4a_pool = tf.keras.layers.Conv2D(32, (1, 1), padding='SAME', activation='relu')(inception_block4a_pool)

inception_block4a = tf.keras.layers.Concatenate()([inception_block4a_1x1, inception_block4a_3x3,
                                                   inception_block4a_5x5, inception_block4a_pool]) # 16x16x240

# inception_block4b Here on Original Inception
# inception_block4c Here on Original Inception
# inception_block4d Here on Original Inception
# inception_block4e Here on Original Inception
max_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='SAME')(inception_block4a) # 8x8x240

inception_block5a_1x1 = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', activation='relu')(max_pool)

inception_block5a_3x3 = tf.keras.layers.Conv2D(80, (1, 1), padding='SAME', activation='relu')(max_pool)
inception_block5a_3x3 = tf.keras.layers.Conv2D(120, (3, 3), padding='SAME', activation='relu')(inception_block5a_3x3)

inception_block5a_5x5 = tf.keras.layers.Conv2D(16, (1, 1), padding='SAME', activation='relu')(max_pool)
inception_block5a_5x5 = tf.keras.layers.Conv2D(64, (5, 5), padding='SAME', activation='relu')(inception_block5a_5x5)

inception_block5a_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(max_pool)
inception_block5a_pool = tf.keras.layers.Conv2D(64, (1, 1), padding='SAME', activation='relu')(inception_block5a_pool)

inception_block5a = tf.keras.layers.Concatenate()([inception_block5a_1x1, inception_block5a_3x3,
                                                   inception_block5a_5x5, inception_block5a_pool]) # 8x8x416
# inception_block5b Here on Original Inception

avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inception_block5a) # 416
dropout = tf.keras.layers.Dropout(0.4)(avg_pool)
out_layer = tf.keras.layers.Dense(10, activation='softmax')(dropout)

model = tf.keras.models.Model(input_layer, out_layer)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=10,
          validation_data=(x_test, y_test))
