import tensorflow as tf


class DenseExpertSum(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseExpertSum, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = tf.expand_dims(
            tf.reduce_sum(h, axis=1), 
            axis=1
        )
        # print(y.shape)
        return y   