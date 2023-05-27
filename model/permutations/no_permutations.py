import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np
from tensorflow.python.keras.utils import control_flow_util 
import scipy

class NoPermutations(tf.keras.layers.Layer):
    """No permutations (identity).
            
    Input:
        An input tensor of shape = (batch_size, ...) # not used

    Output:
        An output tensor of shape = (1, nb_experts, nb_experts)
    """

    def __init__(self,
                 config,
                 task=0,
                 ):
        super(NoPermutations, self).__init__()
        self.task = config["task"]
        self.nb_experts = config["nb_experts"]
#         tf.print("=========self.nb_experts:", self.nb_experts)
        self.no_of_permutations = config["k"]
        
        self.permutation_weights = tf.constant(
            np.array([
                np.identity(self.nb_experts) for _ in range(self.no_of_permutations)
            ]),
            dtype=self.dtype
        )
        tf.print("=========self.permutation_weights.shape:", tf.shape(self.permutation_weights))
                    
    def build(self, input_shape):
        pass
        
    def call(self, inputs):
                
        trace_RRT = tf.linalg.trace(
            tf.matmul(
                self.permutation_weights,
                tf.transpose(self.permutation_weights, perm=[0,2,1])
            )
        )
        trace_RTR = tf.linalg.trace(
            tf.matmul(
                tf.transpose(self.permutation_weights, perm=[0,2,1]),
                self.permutation_weights
            )
        )
        self.add_metric(tf.reduce_mean(trace_RRT), name='trace_RRT')
        self.add_metric(tf.reduce_mean(trace_RTR), name='trace_RTR')

        return self.permutation_weights                    

if __name__ == "__main__":
    config = {
        "nb_experts": 4,
        "k": 2,
    }
    s = NoPermutation(config)
    h = [
        np.random.random((8, 10)) for _ in range(config["nb_experts"])
    ]
    x = np.random.random((8, 5))
    print(s((h,x)).shape)        