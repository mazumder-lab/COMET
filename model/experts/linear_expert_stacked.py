import tensorflow as tf


class LinearExpertStacked(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(LinearExpertStacked, self).__init__()
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"] * config["nb_experts"])
        self.nb_experts = config["nb_experts"]
    def call(
        self,
        x,
        training
    ):
#         tf.print("Expert weights:", self.dense1.weights, summarize=-1)
        y = self.dense1(x)
        y = tf.reshape(y, [y.shape[0], -1, self.nb_experts])
        return y       