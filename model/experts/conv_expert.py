import tensorflow as tf


class ConvExpert(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(ConvExpert, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            config["conv1_nb_filters"],
            config["conv1_kernel_size"],
        )
        self.maxpool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(
            config["conv2_nb_filters"],
            config["conv2_kernel_size"],
        )
        self.maxpool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(
            config["conv3_nb_filters"],
            config["conv3_kernel_size"],
        )
        self.maxpool3 = tf.keras.layers.MaxPooling2D()
        self.conv4 = tf.keras.layers.Conv2D(
            config["conv4_nb_filters"],
            config["conv4_kernel_size"],
        )
        self.maxpool4 = tf.keras.layers.MaxPooling2D()
#         self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
#         self.dense2 = tf.keras.layers.Dense(config["dense2_size"])

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.conv1(x)
        )
        h = self.maxpool1(h)
        h = tf.keras.activations.relu(
            self.conv2(h)
        )
        h = self.maxpool2(h)
        h = tf.keras.activations.relu(
            self.conv3(h)
        )
        h = self.maxpool3(h)
        h = tf.keras.activations.relu(
            self.conv4(h)
        )
        h = self.maxpool4(h)
        # flatten
        y = tf.reshape(h, [h.shape[0], -1])
#         y = tf.keras.activations.relu(
#             self.dense1(h)
#         )
#         y = tf.keras.activations.relu(
#             self.dense2(h)
#         )
        return y  