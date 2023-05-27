import tensorflow as tf


class LinearRegressor(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(LinearRegressor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1)

    def call(
        self,
        x
    ):
        y = self.dense1(x)
        # print(y[0:3])
        return y