import tensorflow as tf


class DenseRegressor(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseRegressor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dense2 = tf.keras.layers.Dense(1)

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = self.dense2(h)
        # print(y[0:3])
        return y