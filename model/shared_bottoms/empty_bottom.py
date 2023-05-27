import tensorflow as tf


class EmptyBottom(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(EmptyBottom, self).__init__()

    def call(
        self,
        x
    ):
        return x  