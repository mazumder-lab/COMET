"""Our implementation of Hash routing

Ref: Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason E Weston. Hash Layers For Large Sparse Models. In NeurIPS 2021.

"""

"""A Keras layer for Hash routing gate."""


import tensorflow as tf
import numpy as np

class HashRoutingGate(tf.keras.layers.Layer):
    """A custom layer for selecting a single expert from a mixture of experts.
    Let f_1, f_2, ..., f_n be the experts. The layer returns:
       f_i(x) where i is the index of the pre-selected expert for sample x.
       g(x) is a one-hot encoding of index i.

    Input:  
        The inputs should be as follows:
            inputs: Tuple of the form: (f, routing_inputs, permutation_weights)
                f: list of experts f_i, each with same shape.
                routing_inputs: 2D tensor of input examples
                permutation_weights: identity or permutation from Permutation-based Local Search.
            training:
                Ignored
            indices:
                one-hot encoding of selected expert index i for each sample.
    
    Output: 
        Tensor, with the same shape as the expert tensors.
    
    """
    def __init__(
        self,
        config,
        task=0,
    ):
        super(HashRoutingGate, self).__init__()
        self.task = config["task"]
        self.nb_experts = config["nb_experts"]
        
    def build(
        self,
        input_shape
    ):
        pass
            
    def call(
        self,
        inputs,
        training,
        indices=None
    ):
        f, x, permutation_weights = inputs
        assert(all([f[i].shape[1] == f[i+1].shape[1] for i in range(len(f)-1)]))
               
        # f: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        # f = [tf.reshape(t, [-1, t.shape[1], 1]) for t in f]
        f = [tf.expand_dims(t, -1) for t in f]
        # f: (bs, dim_exp_i, nb_experts)
        f = tf.concat(f, axis=2)
        
        g = tf.expand_dims(indices, axis=2) # (bs, nb_experts, 1)

        # softmaxes: (bs, nb_experts, 1), perm_mask: [k, nb_experts, nb_experts]
        permutation_weights = tf.reduce_mean(permutation_weights, axis=0) # [nb_experts, nb_experts]
        g_permuted = tf.einsum('bik,ij->bjk', g, permutation_weights) # (bs, nb_experts, 1)
        g_permuted = g_permuted/tf.reduce_sum(g_permuted, axis=1, keepdims=True)  # (b, nb_experts, 1)

        # f: (bs, dim_exp, nb_experts), softmaxes: (bs, nb_experts, 1)    
        # y: (bs, dim_exp)
        y = tf.reshape(
            tf.matmul(
                f,
                g_permuted
            ),
            [-1, f.shape[1]]
        )

        s_concat = tf.where(
            tf.math.less(g_permuted, 1e-5),
            tf.ones_like(g_permuted),
            tf.zeros_like(g_permuted)
        )

        self.add_metric(
            tf.reduce_mean(s_concat),
            name='avg_sparsity'
        )
        soft_averages = tf.reduce_mean(g_permuted, axis=[0]) # (nb_experts,)
        hard_averages = tf.reduce_mean(tf.ones_like(s_concat)-s_concat, axis=[0]) # (nb_experts,)
        soft_averages_for_all_experts_list = tf.split(
            tf.reshape(soft_averages, [-1]),
            self.nb_experts
        )
        [self.add_metric(le, name='soft_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(soft_averages_for_all_experts_list)]
        hard_averages_for_all_experts_list = tf.split(
            tf.reshape(hard_averages, [-1]),
            self.nb_experts
        )
        [self.add_metric(le, name='hard_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(hard_averages_for_all_experts_list)] 

        simplex_constraint = tf.reduce_mean(
            tf.reduce_sum(g_permuted, axis=1),
        )
#             tf.print("========simplex_constraint:", simplex_constraint)
        self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))
        simplex_constraint_fails = tf.reduce_sum(
            tf.reduce_sum(g_permuted, axis=1),
            axis=[1]
        ) # (b, )

        simplex_constraint_fails = tf.where(
            tf.math.less(simplex_constraint_fails, 1.0-1e-5),
            tf.ones_like(simplex_constraint_fails),
            tf.zeros_like(simplex_constraint_fails)
        ) # (b, nb_gates)
        simplex_constraint_fails = tf.reduce_mean(simplex_constraint_fails, axis=0)
        self.add_metric(simplex_constraint_fails, name='simplex_constraint_fails_for_task_{}'.format(self.task+1))

        return y
        
        
    def get_config(self):
        config = super(HashRoutingGate, self).get_config()
        config.update({
            "task": self.task,
            "nb_experts": self.nb_experts
        })
        return config
