import tensorflow as tf
import numpy as np



class TopKSoftmaxGate(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        task=0,
    ):
        super(TopKSoftmaxGate, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        # self.regularization_coef = config[“regularization_coef”]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
        
        if "temperature" in config.keys():
            self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
            self.temperature = (
                tf.constant(config["temperature"]) if config["temperature"] is not None else config["temperature"]
            )
        else:
            self.temperature = None
        if self.use_routing_input:
            self.use_bias = config["use_bias"]

        
    def build(
        self,
        input_shape
    ):
        if not self.use_routing_input:
            self.expert_weights = self.add_weight(
                name="expert_weights",
                shape=(self.nb_experts,)   
            )
        else:
            self.expert_weights = self.add_weight(
                name="expert_weights",
                shape=(self.nb_experts, input_shape[1][1])
            )   
            if self.use_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=(self.nb_experts,),
                    initializer=tf.keras.initializers.Zeros()
                )

            
    def call(
        self,
        inputs,           # inputs = (h,x), h being a list of tensors, all of the same size
        training,
        indices=None
    ):
        h, x, permutation_weights = inputs
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))
               
        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        # h = [tf.reshape(t, [-1, t.shape[1], 1]) for t in h]
        h = [tf.expand_dims(t, -1) for t in h]
        # h: (bs, dim_exp_i, nb_experts)
        h = tf.concat(h, axis=2)
        
        if self.temperature is not None:
            p = tf.shape(self.expert_weights)[0]
            self.iterations.assign_add(1)
            scheduler = (
                1.0-tf.math.exp(
                    -tf.cast(
                        self.temperature, self.expert_weights.dtype
                    ) * tf.cast(
                        self.iterations, dtype=self.expert_weights.dtype
                    )
                )
            )
            k = p - tf.cast(
                tf.math.round( 
                    tf.cast(
                        p - tf.constant(self.k, dtype=p.dtype), 
                        dtype=self.expert_weights.dtype
                    ) * scheduler
                ),
                dtype=p.dtype
            )
            self.add_metric(
                tf.cast(k, dtype=tf.float32),
                name='k'
            )
            # tf.print("k: ", k)
        else:
            k = self.k
        
        if not self.use_routing_input:
            topk = tf.math.top_k(
                tf.reshape(tf.expand_dims(self.expert_weights, 1), [-1]),
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
            softmaxes = tf.nn.softmax(
                tf.expand_dims(topk_prep, 1),  # else, we get an error in the softmax activation
                axis=0
            )
            # y: (bs, dim_exp)
            y = tf.reshape(
                tf.matmul(
                    h,
                    softmaxes # tf.reshape(topk_scattered, [-1, 1])
                ),
                [-1, h.shape[1]]
            )
            
            self.add_metric(
                tf.reduce_mean(
                    tf.where(
                        tf.math.less(softmaxes, 1e-5),
                        tf.ones_like(softmaxes),
                        tf.zeros_like(softmaxes)
                    )
                ),
                name='avg_sparsity'
            )

            return y
        
        else:
            # expert_weights: (bs, nb_experts)
            expert_weights = tf.matmul(
                x,
                tf.transpose(self.expert_weights)
            )

            if self.use_bias:
#                 tf.print("========Bias added", summarize=-1, output_stream=sys.stdout)
                expert_weights += tf.expand_dims(self.bias, axis=0)
            
            # print("expert_weights: ", expert_weights)
            topk = tf.math.top_k(
                expert_weights,
                # k=self.k,
                k=k
            )
            
            num_rows = tf.shape(expert_weights)[0]
            row_range = tf.range(num_rows)
            # row_tensor = tf.tile(row_range[:,None], (1, self.k))
            row_tensor = tf.tile(row_range[:,None], (1, k))
            topk_row_col_indices = tf.stack([row_tensor, topk.indices], axis=2)
            
            topk_scattered = tf.expand_dims(
                tf.scatter_nd(
                    topk_row_col_indices,
                    topk.values,
                    expert_weights.shape
                ),
                -1
            )
            # tf.print("topk scattered: ",topk_scattered, summarize=-1)
            # print("topk: ",topk_scattered)
            # softmaxes = tf.keras.activations.softmax(
            softmaxes = tf.nn.softmax(
                tf.where(
                    tf.math.equal(topk_scattered, tf.constant(0.0)),
                    -np.inf * tf.ones_like(topk_scattered),  # we add the mask here
                    topk_scattered
                ),
                axis=1
            ) # (bs, nb_experts, 1)

#             tf.print("gate softmaxes shape: ",tf.shape(softmaxes), summarize=-1)
#             tf.print("gate softmaxes: ",tf.reduce_sum(tf.reduce_sum(softmaxes, axis=1)), summarize=-1)
            # print("softmaxes: ", softmaxes)
    
            # softmaxes: (bs, nb_experts, 1), perm_mask: [k, nb_experts, nb_experts]
            permutation_weights = tf.reduce_mean(permutation_weights, axis=0) # [nb_experts, nb_experts]
            softmaxes_permuted = tf.einsum('bik,ij->bjk', softmaxes, permutation_weights)
            softmaxes_permuted = softmaxes_permuted/tf.reduce_sum(softmaxes_permuted, axis=1, keepdims=True)  # (b, nb_experts, 1)
            
            # h: (bs, dim_exp, nb_experts), softmaxes: (bs, nb_experts)    
            # y: (bs, dim_exp)
            y = tf.reshape(
                tf.matmul(
                    h,
                    softmaxes_permuted 
                ),
                [-1, h.shape[1]]
            )
            
            s_concat = tf.where(
                tf.math.less(softmaxes_permuted, 1e-5),
                tf.ones_like(softmaxes_permuted),
                tf.zeros_like(softmaxes_permuted)
            )

            self.add_metric(
                tf.reduce_mean(s_concat),
                name='avg_sparsity'
            )
            soft_averages = tf.reduce_mean(softmaxes_permuted, axis=[0]) # (nb_experts,)
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
                tf.reduce_sum(softmaxes_permuted, axis=1),
            )
#             tf.print("========simplex_constraint:", simplex_constraint)
            self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))
            simplex_constraint_fails = tf.reduce_sum(
                tf.reduce_sum(softmaxes_permuted, axis=1),
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
        config = super(TopKSoftmaxGate, self).get_config()
        config.update({
            "k": self.k
        })
        return config
