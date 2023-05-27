import tensorflow as tf
import numpy as np



class HashRoutingGate(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        task=0,
    ):
        super(HashRoutingGate, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        # self.regularization_coef = config[“regularization_coef”]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
        
        
    def build(
        self,
        input_shape
    ):
        pass
            
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
        
        weights = tf.expand_dims(indices, axis=2) # (bs, nb_experts, 1)

#             tf.print("gate softmaxes shape: ",tf.shape(softmaxes), summarize=-1)
#             tf.print("gate softmaxes: ",tf.reduce_sum(tf.reduce_sum(softmaxes, axis=1)), summarize=-1)

        # softmaxes: (bs, nb_experts, 1), perm_mask: [k, nb_experts, nb_experts]
        permutation_weights = tf.reduce_mean(permutation_weights, axis=0) # [nb_experts, nb_experts]
        w_permuted = tf.einsum('bik,ij->bjk', weights, permutation_weights) # (bs, nb_experts, 1)
        w_permuted = w_permuted/tf.reduce_sum(w_permuted, axis=1, keepdims=True)  # (b, nb_experts, 1)

        # h: (bs, dim_exp, nb_experts), softmaxes: (bs, nb_experts, 1)    
        # y: (bs, dim_exp)
        y = tf.reshape(
            tf.matmul(
                h,
                w_permuted
            ),
            [-1, h.shape[1]]
        )

        s_concat = tf.where(
            tf.math.less(w_permuted, 1e-5),
            tf.ones_like(w_permuted),
            tf.zeros_like(w_permuted)
        )

        self.add_metric(
            tf.reduce_mean(s_concat),
            name='avg_sparsity'
        )
        soft_averages = tf.reduce_mean(w_permuted, axis=[0]) # (nb_experts,)
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
            tf.reduce_sum(w_permuted, axis=1),
        )
#             tf.print("========simplex_constraint:", simplex_constraint)
        self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))
        simplex_constraint_fails = tf.reduce_sum(
            tf.reduce_sum(w_permuted, axis=1),
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



if __name__ == '__main__':
    # bs = 5, dim_x = 3
    x = tf.convert_to_tensor(
        [[1., 3., 2.],
        [0., 5., -2.],
        [1., 3., 2.],
        [0., 5., -2.],
        [0., 5., -2.]]
    )
    # nb_exp = 4
    k = 2 
    # nb_exp = 4, dim_x = 3
    expert_weights = tf.convert_to_tensor(
        [[0., 1., -1.],
        [0., 0.5, 1.],
        [1., -0.5, -1.],
        [0., 3., 2.]]
    )

    config = {
        "k": k,
        "use_routing_input": True,
        "nb_experts": 4
    }

    h = [
        np.random.random((5, 1)) for _ in range(config["nb_experts"])
    ]
    gate = TopKSoftmaxGate(config)
    gate.build(([(5,1) for _ in range(config["nb_experts"])], (5, 3)))
    # print(gate.get_weights())
    # gate.set_weights([expert_weights])
    # print(gate.get_weights())

    gate((h,x))

