import tensorflow as tf


class DenseExpertDropout(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseExpertDropout, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dropout1 = tf.keras.layers.Dropout(config["dropout1_rate"])

    def call(
        self,
        x,
        training
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = self.dropout1(h, training=training)
        return y       