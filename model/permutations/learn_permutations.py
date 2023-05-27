import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np
from tensorflow.python.keras.utils import control_flow_util 
import scipy

class LearnPermutations(tf.keras.layers.Layer):
    """Learn (K) permutations.
            
    Input:
        An input tensor of shape = (batch_size, ...) # not used

    Output:
        An output tensor of shape = (K, nb_experts, nb_experts)
    """

    def __init__(self,
                 config,
                 node_index=0,
                 task=0,
                 ):
        super(LearnPermutations, self).__init__()
        self.task = config["task"]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
#         tf.print("=========self.nb_experts:", self.nb_experts)
#         tf.print("=========self.max_depth:", self.max_depth)
        
        self.tau_initial = 1e-3
        self.tau_final = 1e-7
        self.steps_per_epoch = config["steps_per_epoch"]
        self.epochs_for_learning_permutation = config["epochs_for_learning_permutation"]
        self.iterations_for_learning_permutation = tf.constant(
            self.epochs_for_learning_permutation * self.steps_per_epoch
        )
        self.tau_ref = tf.linspace(start=np.log10(self.tau_initial), stop=np.log10(self.tau_final), num=2)
        self.n_iters_ref = tf.linspace(start=20, stop=150, num=2)
        self.learn_k_permutations = config["learn_k_permutations"]
        if self.learn_k_permutations:
            self.no_of_permutations = self.k
        else:
            self.no_of_permutations = 1
        self.noise_factor = config["noise_factor"]
        self.perm_entropy_reg = config["perm_entropy_reg"]
        
        self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')

        self.permutation_log_weights = self.add_weight(
            name="permutation_log_weights",
            shape=(self.no_of_permutations, self.nb_experts, self.nb_experts),
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True
        )
        self.permutation_weights = self.add_weight(
            name="log_weights",
            shape=(self.no_of_permutations, self.nb_experts, self.nb_experts),
            initializer='zeros',
            trainable=False
        )
#         tf.print("=========self.permutation_log_weights.shape:", tf.shape(self.permutation_log_weights))
            
    def build(self, input_shape):
        pass
    
    def _sample_gumbel(self, shape, eps=1e-20):
        """Samples arbitrary-shaped standard gumbel variables.
        Args:
            shape: list of integers
            eps: float, for numerical stability
        Returns:
            A sample of standard Gumbel random variables
        """
        u = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        return -tf.math.log(-tf.math.log(u + eps) + eps)

    def _sinkhorn(self, log_alpha, n_iters=20):
        """Performs incomplete Sinkhorn normalization to log_alpha.
        
        By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
        with positive entries can be turned into a doubly-stochastic matrix
        (i.e. its rows and columns add up to one) via the succesive row and column
        normalization.
        -To ensure positivity, the effective input to sinkhorn has to be
        exp(log_alpha) (elementwise).
        -However, for stability, sinkhorn works in the log-space. It is only at
        return time that entries are exponentiated.
        [1] Sinkhorn, Richard and Knopp, Paul.
        Concerning nonnegative matrices and doubly stochastic
        matrices. Pacific Journal of Mathematics, 1967
        
        Args:
            log_alpha: 2D tensor (a matrix of shape [N, N])
              or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
            n_iters: number of sinkhorn iterations (in practice, as little as 20
              iterations are needed to achieve decent convergence for N~100)
          
        Returns:
            A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
              converted to 3D tensors with batch_size equals to 1)
        """
        b = tf.shape(log_alpha)[0]
        log_alpha = tf.reshape(log_alpha, [b, self.nb_experts, self.nb_experts])
        
        for _ in tf.range(n_iters, dtype=tf.int32):
            log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2), [b, self.nb_experts, 1])
            log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1), [b, 1, self.nb_experts])
        return tf.exp(log_alpha)

    def _gumbel_sinkhorn(
        self,
        log_alpha,
        temp=1.0,
        n_samples=1,
        noise_factor=1.0,
        n_iters=20,
        squeeze=True):
        """Random doubly-stochastic matrices via gumbel noise.
        In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
        a permutation matrix. Therefore, for low temperatures this method can be
        seen as an approximate sampling of permutation matrices, where the
        distribution is parameterized by the matrix log_alpha
        The deterministic case (noise_factor=0) is also interesting: it can be
        shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
        permutation matrix, the solution of the
        matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
        Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
        as approximate solving of a matching problem, otherwise solved via the
        Hungarian algorithm.
        Warning: the convergence holds true in the limit case n_iters = infty.
        Unfortunately, in practice n_iter is finite which can lead to numerical
        instabilities, mostly if temp is very low. Those manifest as
        pseudo-convergence or some row-columns to fractional entries (e.g.
        a row having two entries with 0.5, instead of a single 1.0)
        To minimize those effects, try increasing n_iter for decreased temp.
        On the other hand, too-low temperature usually lead to high-variance in
        gradients, so better not choose too low temperatures.
        Args:
            log_alpha: 2D tensor (a matrix of shape [N, N])
                or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
            temp: temperature parameter, a float.
            n_samples: number of samples
            noise_factor: scaling factor for the gumbel samples. Mostly to explore
                different degrees of randomness (and the absence of randomness, with
                noise_factor=0)
            n_iters: number of sinkhorn iterations. Should be chosen carefully, in
                inverse corresponde with temp to avoid numerical stabilities.
            squeeze: a boolean, if True and there is a single sample, the output will
                remain being a 3D tensor.
        Returns:
            sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
                batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
                squeeze = True then the output is 3D.
            log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
                noisy samples of log_alpha, divided by the temperature parameter. If
                n_samples = 1 then the output is 3D.
        """
        n = tf.shape(log_alpha)[1]
        batch_size = tf.shape(log_alpha)[0]
        log_alpha = tf.reshape(log_alpha, [batch_size, n, n])
        log_alpha_w_noise = tf.tile(log_alpha, [n_samples, 1, 1])
        if noise_factor == 0:
            noise = 0.0
        else:
            noise = self._sample_gumbel([n_samples*batch_size, n, n])*noise_factor
        log_alpha_w_noise += noise
        log_alpha_w_noise /= temp
        sink = self._sinkhorn(log_alpha_w_noise, n_iters)
#         if n_samples > 1 or squeeze is False:
#             sink = tf.reshape(sink, [n_samples, batch_size, n, n])
#             sink = tf.transpose(sink, [1, 0, 2, 3])
#             log_alpha_w_noise = tf.reshape(log_alpha_w_noise, [n_samples, batch_size, n, n])
#             log_alpha_w_noise = tf.transpose(log_alpha_w_noise, [1, 0, 2, 3])
#         return sink, log_alpha_w_noise
        return sink
    
    @tf.function
    def _tf_linear_sum_assignment(self, cost_matrix):
        return tf.numpy_function(func=scipy.optimize.linear_sum_assignment,inp=[cost_matrix],Tout=[tf.int64,tf.int64])
    
    def _generate_mask_per_permutation(self, permutation_weights):
        permutation_weights = tf.squeeze(permutation_weights)
        cost = -permutation_weights
        row_ind, col_ind = self._tf_linear_sum_assignment(cost)
#         tf.print("====col_ind", col_ind)
        permutation_mask = tf.gather(
            tf.eye(tf.shape(permutation_weights)[-1]),
            col_ind,
            axis=0
        )
        return permutation_mask

    def _get_permutation_mask(self, permutation_weights):
        permutation_weights_list = tf.split(permutation_weights, self.no_of_permutations)
        permutation_masks_list = [self._generate_mask_per_permutation(perm) for perm in permutation_weights_list]
        permutation_mask = tf.stack(permutation_masks_list)
#         tf.print("======permutation_mask.shape:", tf.shape(permutation_mask))
        return permutation_mask

#     def _get_permutation_mask(self, permutation_weights):
#         topk = tf.math.top_k(
#             permutation_weights,         
#             k=1
#         )

#         topk_scattered = tf.reduce_sum(
#             tf.one_hot(
#                topk.indices,
#                depth=self.nb_experts
#             ), 
#             axis=2
#         ) 
#         # tf.print("topk_scattered: ", topk_scattered[:3, :3, :10], summarize=-1)

#         # topk_scattered: (k, nb_experts, nb_experts)
#         topk_scattered *= permutation_weights
#         # tf.print("topk_scattered_vals: ", topk_scattered[:3, :3, :10], summarize=-1)

#         # softmaxes: (bs, nb_gates, nb_experts)
#         permutation_weights = tf.nn.softmax(
#             tf.where(
#                 tf.math.equal(topk_scattered, tf.constant(0.0)),
#                 -np.inf * tf.ones_like(topk_scattered),  # we add the mask here
#                 topk_scattered
#             ),
#             axis=-1
#         )
# #         tf.print("iterations:", self.iterations, "permutation_weights (R*R^T): ", tf.matmul(permutation_weights, tf.transpose(permutation_weights, perm=[0,2,1]))[0,:,:], summarize=-1)
# #         tf.print("iterations:", self.iterations, "permutation_weights (R^T*R): ", tf.matmul(tf.transpose(permutation_weights, perm=[0,2,1]), permutation_weights)[0,:,:], summarize=-1)
#         return permutation_weights

    def _compute_permutation_entropy_regularization(
        self,
        permutation_weights,
        eps=1e-6,
    ):
        # Entropy regularization is defined as: sum_{b \in batch_size} sum_{t \in nb_gates} sum_{i \in [k]} -sum_{i=1}^n p_{bti}*log(p_{bti})
        permutation_weights_row_norm = permutation_weights/tf.reduce_sum(permutation_weights, axis=-1, keepdims=True) 

        regularization = tf.reduce_mean(
            tf.math.reduce_sum(
                -permutation_weights * tf.math.log(permutation_weights + eps),
                axis=[1,2]
            )
        ) + tf.reduce_mean(
            tf.math.reduce_sum(
                -permutation_weights_row_norm * tf.math.log(permutation_weights_row_norm + eps),
                axis=[1,2]
            )
        )
        
        return regularization

    def _get_permutation_during_training(
        self,
        permutation_log_weights,
        noise_factor=0.01,
    ):

        log_tau = tfp.math.interp_regular_1d_grid(
            x=tf.cast(self.iterations, dtype=self.dtype),
            x_ref_min=tf.constant(0, dtype=self.dtype),
            x_ref_max=tf.cast(self.iterations_for_learning_permutation, dtype=self.dtype),
            y_ref=tf.cast(self.tau_ref, self.dtype)
        )
        tau = tf.math.pow(tf.constant(10.), log_tau)

        permutation_log_weights = tf.cond(
            tf.math.greater_equal(self.iterations, tf.cast(self.iterations_for_learning_permutation, dtype=self.iterations.dtype)), 
            lambda: tf.stop_gradient(permutation_log_weights),
            lambda: permutation_log_weights
        )
#         tf.print("====permutation_log_weights:", permutation_log_weights)
    
#         permutation_log_weights = tf.reshape(permutation_log_weights, shape=[self.nb_gates*self.k, self.nb_experts, self.nb_experts])
                    
#         permutation_weights = tf.cond(
#             tf.math.greater_equal(self.iterations, tf.cast(self.iterations_for_learning_permutation, dtype=self.iterations.dtype)), 
#             lambda: self._get_permutation_mask(permutation_weights), # hard
#             lambda: permutation_weights # soft
#         )

#         tf.print("===============log_tau:", log_tau)
#         tf.print("===============iterations:", self.iterations, "===tau:", tau)

#         permutation_weights = self._sinkhorn(permutation_log_weights/tau) # soft

        n_iters = tfp.math.interp_regular_1d_grid(
            x=tf.cast(self.iterations, dtype=self.dtype),
            x_ref_min=tf.constant(0, dtype=self.dtype),
            x_ref_max=tf.cast(self.iterations_for_learning_permutation, dtype=self.dtype),
            y_ref=tf.cast(self.n_iters_ref, dtype=self.dtype)
        )
        n_iters = tf.cast(n_iters, dtype=tf.int32)
        
        permutation_weights = self._gumbel_sinkhorn(
            permutation_log_weights,
            temp=tau,
            n_samples=1,
            noise_factor=noise_factor,
            n_iters=n_iters
        )

#         tf.print("===============tau:", tau)
#         tf.print("===============n_iters:", n_iters)
        self.add_metric(tau, name='tau-perm')   
        self.add_metric(log_tau, name='log_tau-perm')   

        self.permutation_weights.assign(permutation_weights)
#         tf.print("====iteration:", self.iterations, "==perm", permutation_weights)
        return permutation_weights

    def _get_permutation_during_inference(
        self,
        permutation_weights,
    ):
        permutation_weights = self._get_permutation_mask(permutation_weights)
#         tf.print("====iteration:", self.iterations, "==perm", permutation_weights)
        return permutation_weights

    def _get_permutation_during_learning_and_after_learning(
        self,
        iterations
    ):
        norm = tf.linalg.norm(
            self._get_permutation_during_inference(self.permutation_weights)-self.permutation_weights,
            axis=[1,2]
        )
#         tf.print("=====norm:", norm)
#         tf.print("=====hard-mask:", self._get_permutation_during_inference(self.permutation_weights))
#         tf.print("=====soft-mask:", self.permutation_weights)
#         tf.print("=====hard-mask (norm):", tf.math.square(tf.linalg.norm(self._get_permutation_during_inference(self.permutation_weights), axis=[1,2])))
#         tf.print("=====soft-mask (norm):", tf.math.square(tf.linalg.norm(self.permutation_weights, axis=[1,2])))
        
        permutation_weights = tf.cond(
            tf.math.less(
                iterations,
                tf.cast(self.iterations_for_learning_permutation, dtype=self.iterations.dtype)
            ),
            lambda: self._get_permutation_during_training(
                self.permutation_log_weights,
                noise_factor=self.noise_factor
            ),
            lambda: self._get_permutation_during_inference(
                self.permutation_weights,
            )
        )
        return permutation_weights

    
    def call(self, inputs, training=True):
        
        increment = control_flow_util.smart_cond(
            training, 
            lambda: tf.ones_like(self.iterations),
            lambda: tf.zeros_like(self.iterations)
        )
        self.iterations.assign_add(increment)
        
        
        permutation_weights = control_flow_util.smart_cond(
            training, 
            lambda: self._get_permutation_during_learning_and_after_learning(self.iterations),
            lambda: self._get_permutation_during_inference(self.permutation_weights)
        )

        explicit_perm_entropy_regularization = control_flow_util.smart_cond(
            training, 
            lambda: self._compute_permutation_entropy_regularization(permutation_weights),
            lambda: tf.zeros(())
        )
        self.add_loss(      
            self.perm_entropy_reg*explicit_perm_entropy_regularization
        ) 
        
#         tf.print("========permutation_weights.shape:", tf.shape(permutation_weights))
#         trace_RRT = tf.linalg.trace(
#             tf.matmul(
#                 permutation_weights,
#                 tf.transpose(permutation_weights, perm=[0,2,1])
#             )
#         )
#         trace_RTR = tf.linalg.trace(
#             tf.matmul(
#                 tf.transpose(permutation_weights, perm=[0,2,1]),
#                 permutation_weights
#             )
#         )
#         self.add_metric(tf.reduce_mean(trace_RRT), name='trace_RRT_task{}'.format(self.task))
#         self.add_metric(tf.reduce_mean(trace_RTR), name='trace_RTR_task{}'.format(self.task))
        
        if not self.learn_k_permutations:
            permutation_weights = tf.tile(permutation_weights  , tf.constant([self.k,1,1], tf.int32))

        return permutation_weights                    

if __name__ == "__main__":
    config = {
        "nb_experts": 4,
        "k": 2
    }
    s = LearnPermutation(config)
    h = [
        np.random.random((8, 10)) for _ in range(config["nb_experts"])
    ]
    x = np.random.random((8, 5))
    print(s((h,x)).shape)        