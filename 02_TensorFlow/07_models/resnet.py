import tensorflow as tf


class ResNet:
    init_channel = 64
    layer_components = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self, input_shape=(None, None, 3), n_classes=0, n_layer=18):
        """

        :param input_shape: (tuple) (h, w, c)
        :param n_classes: If 0 is given, top_layer will not be included.
        :param n_layer: 18, 34, 50, 101, 152
        """
        assert n_layer in list(ResNet.layer_components.keys())
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_layer = n_layer
        self.layer_component = ResNet.layer_components[self.n_layer]

    def get_layer(self, input_layer):
        ResBlock = ResNetBlock if self.n_layer < 50 else BottleNeckBlock

        layer = ConvBN(ResNet.init_channel, (7, 7), strides=(2, 2), name="stem")(input_layer)
        layer = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='SAME', name="max_pool")(layer)

        for i, n_layer in enumerate(self.layer_component):
            channel = ResNet.init_channel * (2**i)

            for j in range(n_layer):
                downsample = True if (i > 0 and j == 0) else False
                layer = ResBlock(channel, (3, 3), downsample=downsample, name=f"resblock_{i}_{j}")(layer)

        if self.n_classes > 0:
            layer = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(layer)
            layer = tf.keras.layers.Dense(self.n_classes, activation='softmax', use_bias=True, name="out_dense")(layer)

        return layer

    def build_model(self):
        input_layer = tf.keras.layers.Input(self.input_shape)
        layer = self.get_layer(input_layer)

        model = tf.keras.models.Model(input_layer, layer)

        return model


class BottleNeckBlock:
    def __init__(self, n_filter, kernel_size, downsample=False, padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="bottle_resblock"):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.padding = padding
        self.activation = activation
        self.name = name
        self.use_bias = use_bias

    def __call__(self, layer):
        strides = (2, 2) if self.downsample else (1, 1)

        x = ConvBN(self.n_filter, (1, 1), strides=(1, 1), padding=self.padding, use_bias=self.use_bias,
                   name=f"{self.name}_conv_bn_front")(layer)

        x = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size, strides=strides, use_bias=self.use_bias,
                                   activation=None, padding=self.padding,
                                   name="{}_{}x{}conv_0".format(self.name, self.kernel_size[0], self.kernel_size[1]))(x)
        if layer.shape[-1] != self.n_filter*4:
            layer = tf.keras.layers.Conv2D(self.n_filter*4, (1, 1), strides=strides, use_bias=self.use_bias,
                                           activation=None, padding=self.padding,
                                           name=f"{self.name}_1x1conv_init")(layer)

        x = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_0")(x)
        x = self.activation(name=f"{self.name}_activation_0")(x)
        x = tf.keras.layers.Conv2D(self.n_filter*4, (1, 1), strides=(1, 1), use_bias=self.use_bias, name=f"{self.name}_1x1conv_1")(x)

        x_layer = tf.keras.layers.Add(name=f"{self.name}_residual")([x, layer])
        x_layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_1")(x_layer)
        x_layer = self.activation(name=f"{self.name}_activation_1")(x_layer)

        return x_layer


class ResNetBlock:
    def __init__(self, n_filter, kernel_size, downsample=False, padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="resblock"):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.padding = padding
        self.activation = activation
        self.name = name
        self.use_bias = use_bias

    def __call__(self, layer):
        strides = (2, 2) if self.downsample else (1, 1)

        x = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size, strides=strides, padding=self.padding,
                                   activation=None, use_bias=self.use_bias,
                                   name="{}_{}x{}conv_0".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)
        if self.downsample:
            layer = tf.keras.layers.Conv2D(self.n_filter, kernel_size=(1, 1), strides=(2, 2),
                                           padding=self.padding, activation=None,
                                           name="{}_{}x{}conv_init".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)

        x = tf.keras.layers.BatchNormalization(name="{}_bn_0".format(self.name))(x)
        x = self.activation(name="{}_activation_0".format(self.name))(x)
        x = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size, strides=(1, 1),
                                   use_bias=self.use_bias, activation=None, padding=self.padding,
                                   name="{}_{}x{}conv_1".format(self.name, self.kernel_size[0], self.kernel_size[1]))(x)

        x_layer = tf.keras.layers.Add(name=f"{self.name}_residual")([x, layer])
        x_layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_1")(x_layer)
        x_layer = self.activation(name=f"{self.name}_activation_1")(x_layer)

        return x_layer


class ConvBN:
    def __init__(self, n_filter, kernel_size, strides=(1, 1), padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="conv_bn"):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

    def __call__(self, layer):
        layer = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size,
                                       strides=self.strides, padding=self.padding,
                                       activation=None, use_bias=self.use_bias,
                                       name="{}_{}x{}conv".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)
        layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")(layer)
        layer = self.activation(name=f"{self.name}_activation")(layer)

        return layer


if __name__ == "__main__":
    resnet = ResNet(input_shape=(224, 224, 3), n_classes=10, n_layer=18)
    model = resnet.build_model()
