""" Contains constraints for projection operations"""
import tensorflow as tf


# def projection_onto_simplex(w):
#     n = tf.shape(w)[0]
#     y = tf.sort(w, direction='DESCENDING')
#     thresh = (tf.cumsum(y)-1.0)/tf.cast(n-tf.range(n-1,-1,-1), dtype=w.dtype)
#     i = tf.searchsorted(thresh-y, tf.zeros(1, dtype=w.dtype))
#     # t = tf.gather(thresh, min(i,n)-1)
#     # print(tf.math.minimum(i,n)-1)
#     t = tf.gather(thresh, tf.math.minimum(i,n)-1)
#     w = tf.maximum(w-t,0)
#     return w

def projection_onto_simplex(w): 
    """Projects (batch-wise) vectors onto a simplex
    
    References:
        - Efficient Projections onto the L1-Ball for Learning in High Dimensions
          [https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf]
        
    Input:
        w: a float tensor of shape (batch, num_experts).
        
    Returns:
        w: a float tensor of shape (batch, num_experts).        
    """
    batch = tf.shape(w)[0]
    n = tf.shape(w)[1]
    y = tf.sort(w, direction='DESCENDING', axis=1)
    thresh = (tf.cumsum(y, axis=1)-1.0)/tf.expand_dims(tf.cast(n-tf.range(n-1,-1,-1), dtype=w.dtype), axis=0)
    
    i = tf.searchsorted(thresh-y, tf.expand_dims(tf.zeros(batch, dtype=w.dtype), axis=1))
    t = tf.gather(thresh, tf.math.minimum(i,n)-1, axis=1, batch_dims=1)
    w = tf.maximum(w-t,0)
    return w


class ProjectionOntoSimplex(tf.keras.constraints.Constraint):
    """Projects weights onto a simplex after gradient update.
    
    Solves the following minimization problem:
        min (1/2)*||y-w||^2 s.t. \sum_i w_i = 1, w_i>=0.
    Returns:
        w: Float Tensor of shape (p,).
    """
    def __init__(self, **kwargs):
        super(ProjectionOntoSimplex, self).__init__(**kwargs)
        pass
    
    def __call__(self, w):
        return projection_onto_simplex(w)   
    
    def get_config(self):
        config = super(ProjectionOntoSimplex, self).get_config()
        return config
    
    
# class SparseProjectionOntoSimplex(tf.keras.constraints.Constraint):
#     """Projects weights onto a simplex after gradient update.
    
#     Solves the following minimization problem:
#         min (1/2)*||y-w||^2 s.t. \sum_i w_i = 1, w_i>=0, ||w||_0<=k
    
#     References:
#         [Sparse projections onto the simplex](https://arxiv.org/pdf/1206.1529.pdf) 
#         by Anastasios Kyrillidis, Stephen Becker, Volkan Cevher, Christoph Koch in ICML

#     Returns:
#         w: Float Tensor of shape (p,).
#     """
#     def __init__(self, k, **kwargs):
#         super(SparseProjectionOntoSimplex, self).__init__(**kwargs)
#         self.k = k
    
#     def __call__(self, w):
#         topk = tf.math.top_k(w, self.k)
#         w_topk = projection_onto_simplex(tf.gather(w, topk.indices))
#         y = tf.scatter_nd(tf.reshape(topk.indices, (-1, 1)), w_topk, tf.shape(w))
#         return y
    
#     def get_config(self):
#         config = super(SparseProjectionOntoSimplex, self).get_config()
#         return config

class SparseProjectionOntoSimplex(tf.keras.constraints.Constraint):
    """Projects (batch-wise) weights onto a simplex after gradient update in a batch-wise fashion.
    
    Solves the following minimization problem:
        min (1/2)*||y-w||^2 s.t. \sum_i w_i = 1, w_i>=0, ||w||_0<=k
    
    References:
        [Sparse projections onto the simplex](https://arxiv.org/pdf/1206.1529.pdf) 
        by Anastasios Kyrillidis, Stephen Becker, Volkan Cevher, Christoph Koch in ICML
        
    Returns:
        w: Float Tensor of shape (p,).
    """
    def __init__(self, k=1, **kwargs):
        super(SparseProjectionOntoSimplex, self).__init__(**kwargs)
        self.k = k
    
    def __call__(self, w):
        topk = tf.math.top_k(w, self.k)
        w_topk = projection_onto_simplex(tf.gather(w, topk.indices, axis=1, batch_dims=1))        
        num_rows = tf.shape(w)[0]
        row_range = tf.range(num_rows)
        row_tensor = tf.tile(row_range[:,None], (1, self.k))
        topk_row_col_indices = tf.stack([row_tensor, topk.indices], axis=2)
        y = tf.scatter_nd(topk_row_col_indices, w_topk, tf.shape(w))
        return y
    
    def get_config(self):
        config = super(SparseProjectionOntoSimplex, self).get_config()
        return config

    
# class ProjectionTrimmedLasso(tf.keras.constraints.Constraint):
#     """Shrinks bottom p-k weights via trimmed lasso after gradient update.
    
#     Solves the following minimization problem:
#         min (1/2)*||y-w||^2 + lam*T_k(y), s.t. y in R^p (simplex = False) or y in Delta_p (simplex=True)
    
#     References:
    
#     Returns:
#         w: Float Tensor of shape (p,).
#     """
#     def __init__(self, k, regularization_coef, learning_rate, temperature=None, do_lam_scheduler=None, simplex=True, **kwargs):
#         super(ProjectionTrimmedLasso, self).__init__(**kwargs)
#         self.k = k
#         # self.lam = regularization_coef * learning_rate 
#         self.regularization_coef = regularization_coef
#         self.learning_rate = learning_rate
#         self.simplex = simplex
#         self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
#         self.temperature = (
#             tf.constant(temperature) if temperature is not None else temperature
#         )
#         self.do_lam_scheduler = (
#             tf.constant(do_lam_scheduler) if do_lam_scheduler is not None else do_lam_scheduler
#         )
    
#     def __call__(self, w):
#         p = tf.shape(w)[0]
#         z = tf.identity(w)
#         if self.temperature is not None:
#             self.iterations.assign_add(1)
#             scheduler = (
#                 1.0-tf.math.exp(
#                     -tf.cast(self.temperature, w.dtype)*tf.cast(self.iterations, dtype=w.dtype)
#                 )
#             )
#             if self.do_lam_scheduler:
#                 lam = self.regularization_coef * self.learning_rate * scheduler
#                 k = self.k
#             else:
#                 lam = self.regularization_coef * self.learning_rate
#                 # print(p.dtype)
#                 # print(tf.constant(self.k, dtype=p.dtype).dtype)
#                 # print((tf.cast(p - tf.constant(self.k, dtype=p.dtype),dtype=w.dtype) * scheduler).dtype)
#                 k = p - tf.cast(
#                     tf.math.round( 
#                         tf.cast(p - tf.constant(self.k, dtype=p.dtype), dtype=w.dtype) * scheduler
#                     ),
#                     dtype=p.dtype
#                 )
#         else:
#             lam = self.regularization_coef * self.learning_rate
#             k = self.k
        
#         if not self.simplex:
#             bottomk_neg = tf.math.top_k(-tf.abs(w), p - k) 
#             # self.bottomk_neg = bottomk_neg.values           
#             z += tf.scatter_nd(
#                 tf.reshape(bottomk_neg.indices, (-1, 1)),
#                 tf.maximum(-bottomk_neg.values-lam, 0.0)*tf.sign(-bottomk_neg.values)-tf.gather_nd(w, tf.reshape(bottomk_neg.indices, (-1, 1))),
#                 tf.shape(w)
#             )
#         else:
#             bottomk_neg = tf.math.top_k(-w, p - k) 

#             # self.bottomk_neg = bottomk_neg.values
#             z += tf.scatter_nd(
#                 tf.reshape(bottomk_neg.indices, (-1, 1)),
#                 tf.zeros_like(bottomk_neg.values) - lam,
#                 tf.shape(w)
#             )
#             z = projection_onto_simplex(z)
#         return z
        
#     def get_config(self):
#         config = super(ProjectionTrimmedLasso, self).get_config()
#         return config

class ProjectionTrimmedLasso(tf.keras.constraints.Constraint):
    """Shrinks (batch-wise) bottom p-k weights via trimmed lasso after gradient update.
    
    Solves the following minimization problem:
        min (1/2)*||y-w||^2 + lam*T_k(y), s.t. y in R^p (simplex = False) or y in Delta_p (simplex=True)
    
    References:
    
    Returns:
        w: Float Tensor of shape (p,).
    """
    def __init__(self, k, regularization_coef, learning_rate, temperature=None, do_lam_scheduler=None, simplex=True, **kwargs):
        super(ProjectionTrimmedLasso, self).__init__(**kwargs)
        self.k = k
        # self.lam = regularization_coef * learning_rate 
        self.regularization_coef = regularization_coef
        self.learning_rate = learning_rate
        self.simplex = simplex
        self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
        self.temperature = (
            tf.constant(temperature) if temperature is not None else temperature
        )
        self.do_lam_scheduler = (
            tf.constant(do_lam_scheduler) if do_lam_scheduler is not None else do_lam_scheduler
        )
    
    def __call__(self, w):
        p = tf.shape(w)[1]
        z = tf.identity(w)
        if self.temperature is not None:
            self.iterations.assign_add(1)
            scheduler = (
                1.0-tf.math.exp(
                    -tf.cast(self.temperature, w.dtype)*tf.cast(self.iterations, dtype=w.dtype)
                )
            )
            if self.do_lam_scheduler:
                lam = self.regularization_coef * self.learning_rate * scheduler
                k = self.k
            else:
                lam = self.regularization_coef * self.learning_rate
                # print(p.dtype)
                # print(tf.constant(self.k, dtype=p.dtype).dtype)
                # print((tf.cast(p - tf.constant(self.k, dtype=p.dtype),dtype=w.dtype) * scheduler).dtype)
                k = p - tf.cast(
                    tf.math.round( 
                        tf.cast(p - tf.constant(self.k, dtype=p.dtype), dtype=w.dtype) * scheduler
                    ),
                    dtype=p.dtype
                )
        else:
            lam = self.regularization_coef * self.learning_rate
            k = self.k
        
        if not self.simplex:
            bottomk_neg = tf.math.top_k(-tf.abs(w), p - k) 
            num_rows = tf.shape(w)[0]
            row_range = tf.range(num_rows)
            row_tensor = tf.tile(row_range[:,None], (1, p - k))
            topk_row_col_indices = tf.stack([row_tensor, bottomk_neg.indices], axis=2)

            z += tf.scatter_nd(
                topk_row_col_indices,
                tf.maximum(
                    -bottomk_neg.values-lam, 0.0
                )*tf.sign(
                    -bottomk_neg.values
                )-tf.gather_nd(
                    w, topk_row_col_indices
                ),
                tf.shape(w)
            )
        else:
            bottomk_neg = tf.math.top_k(-w, p - k) 
            num_rows = tf.shape(w)[0]
            row_range = tf.range(num_rows)
            row_tensor = tf.tile(row_range[:,None], (1, p - k))
            topk_row_col_indices = tf.stack([row_tensor, bottomk_neg.indices], axis=2)

            z += tf.scatter_nd(
                topk_row_col_indices,
                tf.zeros_like(bottomk_neg.values) - lam,
                tf.shape(w)
            )
            z = projection_onto_simplex(z)
        return z
        
    def get_config(self):
        config = super(ProjectionTrimmedLasso, self).get_config()
        return config