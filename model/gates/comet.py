"""This is code for COMET gate

Ref: Shibal Ibrahim, Wenyu Chen, Hussein Hazimeh, Natalia Ponomareva, Zhe Zhao, Rahul Mazumder. COMET: Learning Cardinality Constrained Mixture of Experts with Trees and Local Search. In KDD 2023.

"""

"""A Keras layer for the COMET gate."""

import tensorflow as tf 
import numpy as np
from tensorflow.python.keras.utils import control_flow_util 

# Small constant used to ensure numerical stability.
EPSILON = 1e-6

class SmoothStep(tf.keras.layers.Layer):
    """A smooth-step function.
    For a scalar x, the smooth-step function is defined as follows:
    0                                             if x <= -gamma/2
    1                                             if x >= gamma/2
    3*x/(2*gamma) -2*x*x*x/(gamma**3) + 0.5       o.w.
    See https://arxiv.org/abs/2002.07772 for more details on this function.
    """

    def __init__(self, gamma=1.0):
        """Initializes the layer.
        Args:
          gamma: Scaling parameter controlling the width of the polynomial region.
        """
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2
        self._upper_bound = gamma / 2
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def call(self, inputs):
        return tf.where(
            inputs <= self._lower_bound,
            tf.zeros_like(inputs),
            tf.where(
                inputs >= self._upper_bound,
                tf.ones_like(inputs),
                self._a3 * (inputs**3) + self._a1 * inputs + self._a0
            )
        )


class COMETGate(tf.keras.layers.Layer):
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
        
    Implementation Notes:
        This uses k trees and is a fully vectorized implementation of trees. It treats the ensemble
        of k trees as one "super" tree, where every node stores a dense layer with num_trees units,
        each corresponding to the hyperplane of one tree.
    """

    def __init__(self,
                 config,
                 node_index=0,
                 depth_index=0,
                 name='Node-Root',
                 task=0,
                 ):
        super(COMETGate, self).__init__()
        self.task = config["task"]
        self.nb_experts = config["nb_experts"]
        self.max_depth = (int)(np.ceil(np.log2(self.nb_experts)))
        self.k = config["k"]
        self.node_index = node_index
        self.depth_index = depth_index
        self.max_split_nodes = self.nb_experts - 1
        self.leaf = node_index >= self.nb_experts - 1

        self.gamma = config["gamma"]
        self._z_initializer = config["z_initializer"] or tf.keras.initializers.RandomUniform(
            -self.gamma / 100, self.gamma / 100)
        self._w_initializer = config["w_initializer"] or tf.keras.initializers.RandomUniform()
        self.activation = SmoothStep(self.gamma)
        
        self.entropy_reg = config["entropy_reg"]
            
        if not self.leaf:
            self.selector_layer = tf.keras.layers.Dense(
                self.k,
                use_bias=True,
                activation=self.activation,
                kernel_initializer=self._z_initializer,
                bias_initializer=self._z_initializer, 
            )
            self.left_child = COMETGate(config, 2*self.node_index+1, depth_index=self.depth_index+1, name="Node-"+str(2*self.node_index+1))
            self.right_child = COMETGate(config, 2*self.node_index+2, depth_index=self.depth_index+1, name="Node-"+str(2*self.node_index+2))

        else:
            self.output_layer = tf.keras.layers.Dense(
                self.k,
                use_bias=True,
                activation=None,
                kernel_initializer=self._w_initializer,
                bias_initializer=self._w_initializer,    
                kernel_regularizer=tf.keras.regularizers.L2(self.L2)
            )
                            
    def build(self, input_shape):
        pass
    
    def _compute_entropy_regularization_per_expert(
        self,
        prob,
        entropy_reg,
    ):
        # Entropy regularization is defined as: sum_{b \in batch_size} sum_{i \in [k]} -sum_{i=1}^n p_{bi}*log(p_{bi})
        regularization = entropy_reg * tf.reduce_mean(
            tf.math.reduce_sum(
                -prob * tf.math.log(prob + EPSILON),
                axis=1
            )
        )
        return regularization

    def call(
        self,
        inputs,
        training=True,
        prob=1.0,
        indices=None
    ):
        
        f, x, permutation_weights = inputs
        assert(all([f[i].shape[1] == f[i+1].shape[1] for i in range(len(f)-1)]))

        # f: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        f = [tf.expand_dims(t, -1) for t in f]

        # f: (bs, dim_exp_i, nb_experts)
        f = tf.concat(f, axis=2)
        # tf.print("f concat shape: ", f.shape)
        
        if not self.leaf:
            # shape = (batch_size, k)
            current_prob = self.selector_layer(x) # (batch_size, k)
            s_left_child = self.left_child(inputs, training=training, prob=current_prob * prob)
            s_right_child = self.right_child(inputs, training=training, prob=(1 - current_prob) * prob)
            s_bj = tf.concat([s_left_child, s_right_child], axis=-1)


            if self.node_index==0:
                f = tf.expand_dims(f, axis=2) # (b, dim_exp_i, 1, nb_experts)

                s_bj = tf.reshape(s_bj, shape=[tf.shape(s_bj)[0], -1]) # (b, k*nb_experts)
                g = tf.nn.softmax(s_bj, axis=-1) # (b, k*nb_experts)
                g = tf.reshape(g, shape=[tf.shape(s_bj)[0], self.k, self.nb_experts]) # (b, k, nb_experts)
                g = tf.expand_dims(g, axis=1) # (b, 1, k, nb_experts)

                # g: (b, 1, k, nb_experts), perm_mask: [k, nb_experts, nb_experts]

                g_permuted = tf.einsum('bijk,jkl->bijl', g, permutation_weights)
                g_permuted = tf.reduce_sum(g_permuted, axis=2, keepdims=True) # (b, 1, 1, nb_experts)
                g_permuted = g_permuted/tf.reduce_sum(g_permuted, axis=-1, keepdims=True)  # (b, 1, 1, nb_experts)


                # f:(b, dim_exp_i, 1, nb_experts) * g_permuted: (b, 1, 1, nb_experts)
                y = tf.reduce_sum(f * g_permuted, axis=[2,3]) # (b, dim_exp_i, 1, nb_experts) -> (b, dim_exp_i)

                # Compute s_bj
                s_concat = tf.where(
                    tf.math.less(g_permuted, 1e-5),
                    tf.ones_like(g_permuted),
                    tf.zeros_like(g_permuted)
                ) # (b, 1, 1, nb_experts)
                s_avg = tf.reduce_mean(s_concat, axis=-1) # (b, 1, 1)

                avg_sparsity = tf.reduce_mean(s_avg) # average over batch
                self.add_metric(
                    avg_sparsity,
                    name='avg_sparsity'
                )    
                soft_averages = tf.reduce_mean(g_permuted, axis=[0,1,2]) # (nb_experts,)
                hard_averages = tf.reduce_mean(tf.ones_like(s_concat)-s_concat, axis=[0,1,2]) # (nb_experts,)
                soft_averages_for_all_experts_list = tf.split(
                    tf.reshape(soft_averages, [-1]),
                    self.nb_experts
                )
                [self.add_metric(le, name='soft_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(soft_averages_for_all_experts_list)]
                hard_averages_for_all_experts_list = tf.split(
                    tf.reshape(hard_averages, [-1]),
                    self.nb_experts
                )
#                     [self.add_metric(le, name='hard_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(hard_averages_for_all_experts_list)]   

                simplex_constraint = tf.reduce_mean(
                    tf.reduce_sum(g_permuted, axis=-1),
                )
                self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))

                simplex_constraint_fails = tf.reduce_sum(
                    tf.reduce_sum(g_permuted, axis=-1),
                    axis=[1,2]
                ) # (b, )

                simplex_constraint_fails = tf.where(
                    tf.math.less(simplex_constraint_fails, 1.0-1e-5),
                    tf.ones_like(simplex_constraint_fails),
                    tf.zeros_like(simplex_constraint_fails)
                ) # (b, )
                simplex_constraint_fails = tf.reduce_mean(simplex_constraint_fails, axis=0)
                self.add_metric(simplex_constraint_fails, name='simplex_constraint_fails_for_task_{}'.format(self.task+1))

                return y
            else:
                return s_bj#, s_bj_sp
        else:
            # prob's shape = (b, k)
            # Computing a_bij,    a_bij shape = (b, k)
            a_bij = self.output_layer(x) # (b, k) # Note we do not have access to j as j represents leaves

            prob = tf.expand_dims(prob, axis=-1) # (b, k, 1)
            a_bij = tf.expand_dims(a_bij, axis=-1) # (b, k, 1)                
            log_prob = tf.where(
                tf.math.less_equal(prob, tf.constant(0.0)),
                (tf.experimental.numpy.finfo(tf.float32).min)*tf.ones_like(prob),              
                # tf.math.log(prob+tf.experimental.numpy.finfo(tf.float32).eps)
                tf.math.log(prob+1e-8)
            )
#                 s_bj = tf.reduce_logsumexp(a_bij+log_prob, axis=-1, keepdims=True) # (b, 1)
#                 s_bj_sp = tf.reduce_logsumexp(a_bij+tf.math.log(prob),axis=-1,keepdims=True)
            s_bj = a_bij+log_prob # (b, k, 1) 

            regularization_per_expert = control_flow_util.smart_cond(
                training, 
                lambda: self._compute_entropy_regularization_per_expert(prob, self.entropy_reg),
                lambda: tf.zeros(())
            )
            self.add_loss(      
                regularization_per_expert
            ) 

            return s_bj #,s_bj_sp
