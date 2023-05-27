import tensorflow as tf


class DenseExpertDropoutStacked(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseExpertDropoutStacked, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"] * config["nb_experts"])
        self.dropout1 = tf.keras.layers.Dropout(config["dropout1_rate"])
        self.nb_experts = config["nb_experts"]
    def call(
        self,
        x,
        training
    ):
#         tf.print("Expert weights:", self.dense1.weights, summarize=-1)
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = self.dropout1(h, training=training)
        y = tf.reshape(y, [y.shape[0], -1, self.nb_experts])
        return y       