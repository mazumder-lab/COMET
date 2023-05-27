import tensorflow as tf


class DenseMulticlassClassifier(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseMulticlassClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dense2 = tf.keras.layers.Dense(config["dense2_size"])
        self.dense3 = tf.keras.layers.Dense(config["n_labels"])

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        h = tf.keras.activations.relu(
            self.dense2(h)
        )        
        y = tf.nn.softmax(
            self.dense3(h)
        )
        return y