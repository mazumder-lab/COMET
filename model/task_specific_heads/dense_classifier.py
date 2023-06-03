import tensorflow as tf


class DenseClassifier(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dense2 = tf.keras.layers.Dense(config["n_labels"])

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = tf.nn.sigmoid(
            self.dense2(h)
        )

        return y