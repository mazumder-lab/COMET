import tensorflow as tf


class DenseExpertSumStacked(tf.keras.layers.Layer):

    def __init__(
        self, 
        config
    ):
        super(DenseExpertSumStacked, self).__init__()
        self.nb_experts = config["nb_experts"]
        self.dense1 = tf.keras.layers.Dense(config["dense1_size"] * self.nb_experts)

    def call(
        self,
        x
    ):
        h = tf.keras.activations.relu(
            self.dense1(x)
        )
        
        h = tf.reshape(h, tf.convert_to_tensor([h.shape[0],self.nb_experts,-1], dtype=tf.int64))
        
        # y: (bs, nb_experts) (the size of the output of each expert is 1 so we can squeeze it) Not a good idea to squeeze it, so  affects generality of code. Modified it.
        y = tf.reduce_sum(h, axis=-1, keepdims=True)
        y = tf.transpose(y, perm=(0,2,1))
        return y   