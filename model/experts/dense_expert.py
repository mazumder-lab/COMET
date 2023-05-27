import tensorflow as tf


class DenseExpert(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseExpert, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dense2 = tf.keras.layers.Dense(config["dense2_size"])

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = tf.keras.activations.relu(
            self.dense2(h)
        )
        return y       