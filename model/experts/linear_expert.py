import tensorflow as tf


class LinearExpert(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(LinearExpert, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])

    def call(
        self,
        x,
        training
    ):
        y = self.dense1(x)
        return y       