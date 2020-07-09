import tensorflow as tf


def inception_block(in_layer, n1x1, n3x3r, n3x3, n5x5r, n5x5, npool, reduce=False):
    block_1x1 = tf.keras.layers.Conv2D(n1x1, (1, 1), padding='SAME', activation='relu')(in_layer)

    block_3x3 = tf.keras.layers.Conv2D(n3x3r, (1, 1), padding='SAME', activation='relu')(in_layer)
    block_3x3 = tf.keras.layers.Conv2D(n3x3, (3, 3), padding='SAME', activation='relu')(block_3x3)

    block_5x5 = tf.keras.layers.Conv2D(n5x5r, (1, 1), padding='SAME', activation='relu')(in_layer)
    block_5x5 = tf.keras.layers.Conv2D(n5x5, (5, 5), padding='SAME', activation='relu')(block_5x5)

    block_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='SAME')(in_layer)
    block_pool = tf.keras.layers.Conv2D(npool, (1, 1), padding='SAME', activation='relu')(block_pool)

    block = tf.keras.layers.Concatenate()([block_1x1, block_3x3, block_5x5, block_pool])

    if reduce:
        block = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='SAME')(block)

    return block


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

input_layer = tf.keras.layers.Input(shape=(32, 32, 3))

stem_layer = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='SAME', activation='relu')(input_layer)
# Max Pooling Here on Original Inception (3x3, stride=2)
stem_layer = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='SAME', activation='relu')(stem_layer)
# Max Pooling Here on Original Inception (3x3, stride=2)

inception_block3a = inception_block(stem_layer, 32, 48, 64, 8, 16, 16, reduce=False) # 32x32x128
inception_block3b = inception_block(inception_block3a, 64, 64, 96, 16, 48, 32, reduce=True) # 16x16x240

inception_block4a = inception_block(inception_block3b, 96, 48, 104, 8, 24, 32, reduce=False) # 16x16x256
inception_block4b = inception_block(inception_block4a, 80, 56, 112, 12, 32, 32, reduce=False) # 16x16x256
inception_block4c = inception_block(inception_block4b, 64, 64, 128, 12, 32, 32, reduce=False) # 16x16x256
inception_block4d = inception_block(inception_block4c, 56, 72, 144, 16, 32, 32, reduce=False) # 16x16x264
inception_block4e = inception_block(inception_block4d, 128, 80, 160, 16, 64, 64, reduce=True) # 8x8x416

inception_block5a = inception_block(inception_block4e, 128, 80, 160, 16, 64, 64, reduce=False) # 8x8x416
inception_block5b = inception_block(inception_block5a, 192, 96, 192, 24, 64, 64, reduce=False) # 8x8x512

avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inception_block5b) # 512
dropout = tf.keras.layers.Dropout(0.4)(avg_pool)
out_layer = tf.keras.layers.Dense(10, activation='softmax')(dropout)

model = tf.keras.models.Model(input_layer, out_layer)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=10,
          validation_data=(x_test, y_test))
