import tensorflow as tf


class DenseBottom(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseBottom, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dense2 = tf.keras.layers.Dense(config["dense2_size"])

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
        # h = tf.nn.relu(
            self.dense1(x)
        )
        y = tf.keras.activations.relu(
        # y = tf.nn.relu(
            self.dense2(h)
        )
        return y    