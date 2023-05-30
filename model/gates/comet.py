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
    """An ensemble of soft decision trees.
    
    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.
    
    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with 
        num_trees units, each corresponding to the hyperplane of one tree.
    
    Input:
        An input tensor of shape = (batch_size, ...)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
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
        self.use_routing_input = config["use_routing_input"]
        self.nb_experts = config["nb_experts"]
        self.max_depth = (int)(np.ceil(np.log2(self.nb_experts)))
        self.k = config["k"]
        self.L2 = config["L2"]
#         tf.print("=========self.nb_experts:", self.nb_experts)
#         tf.print("=========self.max_depth:", self.max_depth)
        self.node_index = node_index
#         tf.print("=========self.node_index:", self.node_index)
        self.depth_index = depth_index
#         self.max_split_nodes = 2**self.max_depth - 1
        self.max_split_nodes = self.nb_experts - 1
#         self.leaf = node_index >= self.max_split_nodes
        self.leaf = node_index >= self.nb_experts - 1
#         assert self.nb_experts == 2**self.max_depth # to check number of experts is a power of 2 

        self.gamma = config["gamma"]
        self._z_initializer = config["z_initializer"] or tf.keras.initializers.RandomUniform(
            -self.gamma / 100, self.gamma / 100)
        self._w_initializer = config["w_initializer"] or tf.keras.initializers.RandomUniform()
        self.activation = SmoothStep(self.gamma)
        
        self.entropy_reg = config["entropy_reg"]
        if "temperature" in config.keys():
            self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
            self.temperature = (
                tf.constant(config["temperature"]) if config["temperature"] is not None else config["temperature"]
            )
        else:
            self.temperature = None
            
        if not self.use_routing_input:
            if not self.leaf:
                self.selector_weights = self.add_weight(
                    name="selector_weights",
                    shape=(1, self.k),
                    initializer=self._z_initializer,
                    trainable=True
                )
                self.left_child = COMETGate(config, 2*self.node_index+1, depth_index=self.depth_index+1, name="Node-"+str(2*self.node_index+1))
                self.right_child = COMETGate(config, 2*self.node_index+2, depth_index=self.depth_index+1, name="Node-"+str(2*self.node_index+2))
            else:
                masking = np.zeros((1, 1, self.nb_experts))
#                 tf.print("=========self.nb_experts:", self.nb_experts)
#                 tf.print("=========self.node_index:", self.node_index)
#                 tf.print("=========self.max_split_nodes:", self.max_split_nodes)
#                 tf.print("=========self.node_index-self.max_split_nodes:", self.node_index-self.max_split_nodes)
                masking[:,:,self.node_index-self.max_split_nodes] = masking[:,:, self.node_index-self.max_split_nodes] + 1
                self.masking = tf.constant(masking, dtype=self.dtype)
                self.output_weights = self.add_weight(
                    name="output_weights",
                    shape=(1, self.k),
                    initializer=self._w_initializer,
                    trainable=True
                )
        else:
            if not self.leaf:
                self.selector_layer = tf.keras.layers.Dense(
                    self.k,
                    use_bias=True,
                    activation=self.activation,
                    kernel_initializer=self._z_initializer,
                    bias_initializer=self._z_initializer, 
                    kernel_regularizer=tf.keras.regularizers.L2(self.L2)
                )
                self.left_child = COMETGate(config, 2*self.node_index+1, depth_index=self.depth_index+1, name="Node-"+str(2*self.node_index+1))
                self.right_child = COMETGate(config, 2*self.node_index+2, depth_index=self.depth_index+1, name="Node-"+str(2*self.node_index+2))
                if self.balanced_splitting:
                    self.alpha_ave_past = self.add_weight(
                        name="alpha_ave",
                        shape=(1, self.k),
                        initializer='zeros',
                        trainable=False
                    )

            else:
                self.output_layer = tf.keras.layers.Dense(
                    self.k,
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self._w_initializer,
                    bias_initializer=self._w_initializer,    
                    kernel_regularizer=tf.keras.regularizers.L2(self.L2)
                )
                
    #             self.leaf_weight = self.add_weight(shape=[1, self.leaf_dims, self.num_trees], trainable=True, name="Node-"+str(self.node_index))
            
    def build(self, input_shape):
        pass

#     def _compute_weights(self, s_left_child, s_right_child, d_left_child, d_right_child):
#         s_bj_concat = tf.concat([s_left_child, s_right_child], axis=-1) # (b, 2)
#         s_bj = tf.math.reduce_logsumexp(s_bj_concat, axis=-1, keepdims=True) # (b, 1)
#         s_bj_normalized = tf.nn.softmax(s_bj_concat, axis=-1) # (b, 2)

#         s_left_child_normalized, s_right_child_normalized = tf.split(s_bj_normalized, num_or_size_splits=2, axis=-1)
#         d_bj = tf.concat([
#             s_left_child_normalized*d_left_child,
#             s_right_child_normalized*d_right_child
#         ], axis=-1) # (b, j)
#         return s_bj, d_bj

    
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
        
        h, x, permutation_weights = inputs
        
        # tf.print("\ninput of softmax gate: ",len(h), h[0].shape, x.shape)
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))

        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        h = [tf.expand_dims(t, -1) for t in h]

        # h: (bs, dim_exp_i, nb_experts)
        h = tf.concat(h, axis=2)
        # tf.print("h concat shape: ", h.shape)
        
        if not self.use_routing_input:
            if not self.leaf:
                # shape = (batch_size, k)
                current_prob = self.activation(self.selector_weights) # (b=1, k)
                f_left_child, w_left_child = self.left_child(inputs, training=training, prob=current_prob * prob)
                f_right_child, w_right_child = self.right_child(inputs, training=training, prob=(1 - current_prob) * prob)
                f_agg = f_left_child + f_right_child
                w_concat = tf.concat([w_left_child, w_right_child], axis=-1)
                if self.node_index==0:
                    w_agg = tf.reduce_sum(w_concat, axis=-1)
                    y_agg = f_agg/w_agg
                    w_concat = w_concat/tf.reduce_sum(w_concat, axis=-1, keepdims=True)  # (b=1, 1, nb_experts)
                    # Compute s_bj
                    s_concat = tf.where(
                        tf.math.less(w_concat, 1e-5),
                        tf.ones_like(w_concat),
                        tf.zeros_like(w_concat)
                    ) # (b, 1, nb_experts)
                    s_avg = tf.reduce_mean(s_concat, axis=-1) # (b, 1)
#                     tf.print("======s_avg.shape:", s_avg.shape)

                    avg_sparsity = tf.reduce_mean(s_avg) # average over batch
                    self.add_metric(
                        avg_sparsity,
                        name='avg_sparsity'
                    )    
                    return y_agg
                else:
                    return f_agg, w_concat
            else:
                # prob's shape = (b=1, k)
                # Computing a_bij,    a_bij shape = (b=1, k)
                a_bij = self.output_weights # (b=1, k) # Note we do not have access to j as j represents leaves
                a_exp_bij = tf.math.exp(a_bij)
                r_bij = prob * a_exp_bij # (b=1, k)
                r_bij = tf.expand_dims(r_bij, axis=1) # (b=1, 1, k)
                
                # Computing w_bj, 
                w_bj = tf.reduce_sum(r_bij, axis=-1) # (b=1, 1, k) -> (b=1, 1)
                w_bj = tf.expand_dims(w_bj, axis=-1) # (b=1, 1, 1)
#                 tf.print("======w_bj.shape: ", w_bj.shape)
                
                # Computing f_bj
                # Get output of specific expert:  h:(b, dim_exp_i, nb_experts), self.masking: (1, 1, self.nb_experts)
                h_bj = tf.reduce_sum(h * self.masking, axis=-1, keepdims=True) # (b, dim_exp_i, 1)
                f_bij = r_bij * h_bj # (b, dim_exp_i, k)
                f_bj = tf.reduce_sum(f_bij, axis=-1) # (b, dim_exp_i, k) -> (b, dim_exp_i) 
#                 tf.print("======f_bj.shape: ", f_bj.shape)
                
                regularization_per_expert = control_flow_util.smart_cond(
                    training, 
                    lambda: self._compute_entropy_regularization_per_expert(prob, self.entropy_reg),
                    lambda: tf.zeros(())
                )
                self.add_loss(      
                    regularization_per_expert
                ) 

                return f_bj, w_bj
        else:
            if not self.leaf:
                # shape = (batch_size, k)
                current_prob = self.selector_layer(x) # (batch_size, k)
                s_left_child = self.left_child(inputs, training=training, prob=current_prob * prob)
                s_right_child = self.right_child(inputs, training=training, prob=(1 - current_prob) * prob)
                s_bj = tf.concat([s_left_child, s_right_child], axis=-1)
#                 s_bj_sp = tf.concat([s_left_child_sp, s_right_child_sp], axis=-1)  


                if self.node_index==0:
                    h = tf.expand_dims(h, axis=2) # (b, dim_exp_i, 1, nb_experts)
#                     tf.print("======h.shape: ", tf.shape(h))        
        
                    s_bj = tf.reshape(s_bj, shape=[tf.shape(s_bj)[0], -1]) # (b, k*nb_experts)
                    s_bj = tf.nn.softmax(s_bj, axis=-1) # (b, k*nb_experts)
                    w_concat = tf.reshape(s_bj, shape=[tf.shape(s_bj)[0], self.k, self.nb_experts]) # (b, k, nb_experts)
                    w_concat = tf.expand_dims(w_concat, axis=1) # (b, 1, k, nb_experts)

                    # w_concat: (b, 1, k, nb_experts), perm_mask: [k, nb_experts, nb_experts]

                    w_permuted = tf.einsum('bijk,jkl->bijl', w_concat, permutation_weights)
                    w_permuted = tf.reduce_sum(w_permuted, axis=2, keepdims=True) # (b, 1, 1, nb_experts)
                    w_permuted = w_permuted/tf.reduce_sum(w_permuted, axis=-1, keepdims=True)  # (b, 1, 1, nb_experts)
                    

                    # h:(b, dim_exp_i, 1, nb_experts) * w_permuted: (b, 1, 1, nb_experts)
                    y_agg = tf.reduce_sum(h * w_permuted, axis=[2,3]) # (b, dim_exp_i, 1, nb_experts) -> (b, dim_exp_i)

                    # Compute s_bj
                    s_concat = tf.where(
                        tf.math.less(w_permuted, 1e-5),
                        tf.ones_like(w_permuted),
                        tf.zeros_like(w_permuted)
                    ) # (b, 1, 1, nb_experts)
                    s_avg = tf.reduce_mean(s_concat, axis=-1) # (b, 1, 1)

                    avg_sparsity = tf.reduce_mean(s_avg) # average over batch
                    self.add_metric(
                        avg_sparsity,
                        name='avg_sparsity'
                    )    
                    soft_averages = tf.reduce_mean(w_permuted, axis=[0,1,2]) # (nb_experts,)
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
                        tf.reduce_sum(w_permuted, axis=-1),
                    )
                    self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))

                    simplex_constraint_fails = tf.reduce_sum(
                        tf.reduce_sum(w_permuted, axis=-1),
                        axis=[1,2]
                    ) # (b, )

                    simplex_constraint_fails = tf.where(
                        tf.math.less(simplex_constraint_fails, 1.0-1e-5),
                        tf.ones_like(simplex_constraint_fails),
                        tf.zeros_like(simplex_constraint_fails)
                    ) # (b, )
                    simplex_constraint_fails = tf.reduce_mean(simplex_constraint_fails, axis=0)
                    self.add_metric(simplex_constraint_fails, name='simplex_constraint_fails_for_task_{}'.format(self.task+1))
                    
                    return y_agg, soft_averages, hard_averages
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

                return s_bj#,s_bj_sp
