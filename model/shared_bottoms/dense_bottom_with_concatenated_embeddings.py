import tensorflow as tf


class DenseBottomWithConcatenatedEmbeddings(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseBottomWithConcatenatedEmbeddings, self).__init__()
        self.embedding1 = tf.keras.layers.Embedding(
            config["embedding1_vocab_size"],
            config["embedding1_size"]
        )
        self.embedding2 = tf.keras.layers.Embedding(
            config["embedding2_vocab_size"],
            config["embedding2_size"]
        )
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"])
        self.dense2 = tf.keras.layers.Dense(config["dense2_size"])

    def call(
        self,
        x
    ):
        # x is composed of pairs of indices: (bs, 2)
        # e1: (bs, embedding1_size)
        e1 = self.embedding1(
            x[:,0]
            # tf.reshape(x[:,0], [-1, 1])
        )
        # e2: (bs, embedding2_size)
        e2 = self.embedding2(
            x[:,1]
            # tf.reshape(x[:,1], [-1, 1])
        )
        x = tf.concat(
            [e1, e2],
            axis=1
        )
        # print(tf.shape(x))
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        y = tf.keras.activations.relu(
            self.dense2(h)
        )
        return y   