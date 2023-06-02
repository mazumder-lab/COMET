"""Our implementation of Top-k routing

Ref: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In ICLR 2017.

The implementation is based on the description in Section 2.1.
"""

"""A Keras layer for Top-k routing gate."""

import tensorflow as tf
import numpy as np



class TopkGate(tf.keras.layers.Layer):
    """A custom layer for selecting a sparse mixture of experts.
    Let f_1, f_2, ..., f_n be the experts. The layer returns:
              g_1 * f_1 + g_2 * f_2 + ... + g_n * f_n,
    where the mixture weights satisfy:
        (1) cardinality constraint: ||g||_0 <= k
        (2) simplex constraint: g_1, ..., g_n >= 0 and g_1 + ... + g_n = 1.
    The number of non-zeros in the mixture weights can be directly controlled.
    The layer is trained using first-order methods like SGD.
    
    Input:
        The inputs should be as follows:
            inputs: Tuple of the form: (f, routing_inputs, permutation_weights)
                f: list of experts f_i, each with same shape.
                routing_inputs: 2D tensor of input examples
                permutation_weights: identity or permutation from Permutation-based Local Search.
            training: 
                Ignored
            indices:
                Ignored
        
    Output:
        Tensor, with the same shape as the expert tensors.
    """
    def __init__(
        self,
        config,
        task=0,
    ):
        super(TopkGate, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
        self.jitter = config["jitter"]
        
    def build(
        self,
        input_shape
    ):
        if not self.use_routing_input:
            self.gate_weights = self.add_weight(
                name="gate_weights",
                shape=(self.nb_experts,)   
            )
        else:
            self.gate_weights = self.add_weight(
                name="gate_weights",
                shape=(self.nb_experts, input_shape[1][1])
            )   
            self.bias = self.add_weight(
                name="bias",
                shape=(self.nb_experts,),
                initializer=tf.keras.initializers.Zeros()
            )
            if self.jitter:
                self.jitter_weights = self.add_weight(
                    name="jitter_weights",
                    shape=(self.nb_experts, input_shape[1][1])
                )

            
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
        
        k = self.k
        
        if not self.use_routing_input:
            topk = tf.math.top_k(
                tf.reshape(tf.expand_dims(self.gate_weights, 1), [-1]),
                # k=self.k,
                k=k,
            )
            topk_scattered = tf.scatter_nd(
                tf.reshape(topk.indices, [-1, 1]),
                topk.values,
                [self.nb_experts]
            )
            topk_prep = tf.where(
                tf.math.equal(topk_scattered, tf.constant(0.0)),
                -np.inf * tf.ones_like(topk_scattered),  # we add the mask here
                topk_scattered
            )
            # softmaxes: (nb_experts, 1)
            g = tf.nn.softmax(
                tf.expand_dims(topk_prep, 1),  # else, we get an error in the softmax activation
                axis=0
            )
            # y: (bs, dim_exp)
            y = tf.reshape(
                tf.matmul(
                    f,
                    g # tf.reshape(topk_scattered, [-1, 1])
                ),
                [-1, f.shape[1]]
            )
            
            self.add_metric(
                tf.reduce_mean(
                    tf.where(
                        tf.math.less(g, 1e-5),
                        tf.ones_like(g),
                        tf.zeros_like(g)
                    )
                ),
                name='avg_sparsity'
            )

            return y
        
        else:
            # gate_weights: (bs, nb_experts)
            gate_logits = tf.matmul(
                x,
                tf.transpose(self.gate_weights)
            )+tf.expand_dims(self.bias, axis=0)
            
            if self.jitter and training:
                gate_logits += tf.random.normal(gate_logits.shape)*tf.keras.activations.softplus(
                    tf.matmul(
                        x,
                        tf.transpose(self.jitter_weights)
                    )
                )
                
            # print("gate_weights: ", gate_weights)
            topk = tf.math.top_k(
                gate_logits,
                # k=self.k,
                k=k
            )
            
            num_rows = tf.shape(gate_logits)[0]
            row_range = tf.range(num_rows)
            # row_tensor = tf.tile(row_range[:,None], (1, self.k))
            row_tensor = tf.tile(row_range[:,None], (1, k))
            topk_row_col_indices = tf.stack([row_tensor, topk.indices], axis=2)
            
            topk_scattered = tf.expand_dims(
                tf.scatter_nd(
                    topk_row_col_indices,
                    topk.values,
                    gate_logits.shape
                ),
                -1
            )
            
            g = tf.nn.softmax(
                tf.where(
                    tf.math.equal(topk_scattered, tf.constant(0.0)),
                    -np.inf * tf.ones_like(topk_scattered),  # we add the mask here
                    topk_scattered
                ),
                axis=1
            ) # (bs, nb_experts, 1)
    
            # g: (bs, nb_experts, 1), perm_mask: [k, nb_experts, nb_experts]
            permutation_weights = tf.reduce_mean(permutation_weights, axis=0) # [nb_experts, nb_experts]
            g_permuted = tf.einsum('bik,ij->bjk', g, permutation_weights)
            g_permuted = g_permuted/tf.reduce_sum(g_permuted, axis=1, keepdims=True)  # (b, nb_experts, 1)
            
            # f: (bs, dim_exp, nb_experts), g: (bs, nb_experts)    
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
            
            return y, soft_averages, hard_averages
        
        
    def get_config(self):
        config = super(TopkGate, self).get_config()
        config.update({
            "k": self.k
        })
        return config