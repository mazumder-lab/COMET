import tensorflow as tf 
import numpy as np


class SoftmaxGate(tf.keras.layers.Layer):

    def __init__(
        self,
        config,
    ):
        super(SoftmaxGate, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        # self.regularization_coef = config["regularization_coef"]
        self.nb_experts = config["nb_experts"]
        if self.use_routing_input:
            self.use_bias = config["use_bias"]


    def build(
        self,
        input_shape
    ):
        if not self.use_routing_input:
            self.expert_weights = self.add_weight(
                name="expert_weights",
                shape=(self.nb_experts,),      # shapes need to be in this order otherwise topk in the constraint won't work
                # initializer=tf.keras.initializers.HeNormal()
            )
        else:
            self.expert_weights = self.add_weight(
                name="expert_weights",
                shape=(self.nb_experts, input_shape[1][1]),     # input_shape[1][1] corresponds to the shape of each input vector
                # initializer=tf.keras.initializers.HeNormal()
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
        # tf.print("\ninput of softmax gate: ",len(h), h[0].shape, x.shape)
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))

        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        h = [tf.expand_dims(t, -1) for t in h]
        
        # h: (bs, dim_exp_i, nb_experts)
        h = tf.concat(h, axis=2)
        # tf.print("h concat shape: ", h.shape)
        
        if not self.use_routing_input:
            # TODO: GO BACK TO OLD VERSION
            #stable_logits = self.expert_weights - tf.expand_dims(
            #    tf.reduce_max(self.expert_weights),
            #    -1
            #)
            
            # softmaxes: (nb_experts, 1)
            #softmaxes = tf.keras.activations.softmax(
            #    tf.expand_dims(self.expert_weights, 1), 
            #    #tf.expand_dims(stable_logits, 1),       # TODO: GO BACK TO OLD VERSION
            #    axis=1  
            #)
            # tf.print(self.expert_weights.shape)
            # softmaxes = tf.keras.activations.softmax(
            softmaxes = tf.nn.softmax(
                tf.expand_dims(self.expert_weights, 1),       # TODO: GO BACK TO OLD VERSION
                axis=0                    
            )
            
            # tf.print("gate stable_logits: ",stable_logits, summarize=-1)
            # tf.print("gate stable_logits: ",self.expert_weights, summarize=-1)
            # tf.print("gate softmaxes: ",softmaxes, summarize=-1)
            
            # y: (bs, dim_exp)
            
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
            
            y = tf.reshape(
                tf.matmul(
                    h,
                    softmaxes
                ),
                [-1, h.shape[1]]
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
                

            # TODO: GO BACK TO OLD VERSION
            #stable_logits = expert_weights - tf.expand_dims(
            #    tf.reduce_max(expert_weights),
            #    -1
            #)
            
            # softmaxes: (bs, nb_experts, 1)
            softmaxes = tf.expand_dims(
                tf.nn.softmax(          
                    expert_weights,        # TODO: GO BACK TO OLD VERSION
                    axis=1
                ),
                -1
            )
            # tf.print("gate softmaxes: ",softmaxes, summarize=-1)

            permutation_weights = tf.reduce_mean(permutation_weights, axis=0) # [nb_experts, nb_experts]
            softmaxes_permuted = tf.einsum('bik,ij->bjk', softmaxes, permutation_weights)
            
            # print("softmax shape: ",softmaxes.shape)
            # y: (bs, dim_exp)
            y = tf.reshape(
                tf.matmul(
                    h,
                    softmaxes_permuted
                ),
                [-1, h.shape[1]]
            )
            # print("output shape: ",y.shape)
            
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
