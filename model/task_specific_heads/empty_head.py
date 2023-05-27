import tensorflow as tf


class EmptyHead(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(EmptyHead, self).__init__()

    def call(
        self,
        x
    ):
        return x