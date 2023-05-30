# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Keras layer for the dSelect-K MoE gate."""

import math
import sys
from typing import Callable, List, Optional, Tuple, Union
from tensorflow.python.keras.utils import control_flow_util 

import numpy as np
import tensorflow as tf

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

  def __init__(self, gamma = 1.0):
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
        inputs <= self._lower_bound, tf.zeros_like(inputs),
        tf.where(inputs >= self._upper_bound, tf.ones_like(inputs),
                 self._a3 * (inputs**3) + self._a1 * inputs + self._a0))


class EntropyRegularizer(tf.keras.layers.Layer):
  """A regularizer that minimizes entropy.
  This class maintains a counter:
    num_calls = num_calls_per_training_step * num_training_steps
  to allow the regularization parameter to change over the course of training.
  """

  def __init__(
      self,
      schedule_fn = lambda x: 1e-6):
    """Initializes the layer.
    Args:
      schedule_fn: A callable that returns the regularization parameter based on
        the num_calls counter.
    """
    super(EntropyRegularizer, self).__init__()
    self._num_calls = self.add_weight(
        name="num_calls", shape=[], initializer="zeros", trainable=False)
    self._schedule_fn = schedule_fn

  def call(self, inputs):
    assign_op = self._num_calls.assign_add(1, read_value=False)

    preconditions = [] if assign_op is None else [assign_op]
    with tf.control_dependencies(preconditions):
      reg_param = self._schedule_fn(self._num_calls)
      # print("shape : ",(inputs * tf.math.log(inputs + EPSILON)).shape)
      if len(tuple(inputs.shape)) < 3:
        inputs = tf.expand_dims(inputs, 0)

      entropy = -tf.math.reduce_mean(
        tf.math.reduce_sum(
          inputs * tf.math.log(inputs + EPSILON),
          axis=[1,2]
        )
      )
      # order of magnitude changes way too much with the following: (bc no averaging over batch)
      # entropy = -tf.math.reduce_sum(
      #   inputs * tf.math.log(inputs + EPSILON),
      # )
      return reg_param * entropy

class DSelectKWrapperGate(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
    ):
        super(DSelectKWrapperGate, self).__init__()
        print(config)
        self.use_routing_input = config["use_routing_input"]
        # self.use_in_training = config["use_in_training"]
        self.gate = DSelectKGate(
          config["k"],
          config["gamma"],
          EntropyRegularizer(lambda x: config["entropy_reg"]),
          config["z_initializer"],
          config["w_initializer"],
          config["zeta"],
          config.get("load_balancing_penalty", 0.),
          config.get("exp_decay_mov_ave", 0.),
          config["task"]
        )

    def call(
        self,
        inputs,           # inputs = (h,x), h being a list of tensors, all of the same size
        training=True,
        indices=None
    ):
        h, x, _ = inputs
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))

        if not self.use_routing_input:
          y = self.gate(h, training=training)
          return y
        else:
          y = self.gate(inputs, training=training)
          return y

class DSelectKGate(tf.keras.layers.Layer):
    """A custom layer for selecting a sparse mixture of experts.
    Let f_1, f_2, ..., f_n be the experts. The layer returns:
              a_1 * f_1 + a_2 * f_2 + ... + a_n * f_n,
    where the mixture weights satisfy a_1, ..., a_n >= 0 and a_1 + ... + a_n = 1.
    The number of non-zeros in the mixture weights can be directly controlled.
    The layer is differentiable and can be trained using first-order methods like
    SGD.
    Input: For task-only conditioning, the input should be a list of tensors,
    each corresponding to an expert. For example conditioning, the input should
    be a tuple of the form: (experts, routing_inputs), where experts is a list
    of expert tensors, and routing_inputs is a 2D tensor of input examples.
    Note: In both cases, the expert tensors should have the same shape.
    Output: Tensor, with the same shape as the expert tensors.
    Example:
    # Construct a DSelectKGate to select 2 out of 4 experts.
    gate = DSelectKGate(num_nonzeros=2)
    # output_tensor is a sparse mixture of the 4 tensors in the inputs.
    output_tensor = gate(inputs)
    """
    def __init__(
        self,
        num_nonzeros,
        gamma=1.0,
        entropy_reg=None,
        z_initializer=None,
        w_initializer=None,
        zeta=1.0,
        load_balancing_penalty=0.,
        exp_decay_mov_ave=0.,
        task=0,
    ):
        """DSelectKGate constructor.
        Args:
          num_nonzeros: An upper bound on the number of non-zeros in the mixture
            weights (a_1, ..., a_n), where n is the number of experts inferred from
            the inputs to this layer.
          gamma: A scaling parameter for the smooth-step function.
          entropy_reg: An optional `EntropyRegularizer`. This regularizer, if set,
            can be used to control the speed of convergence of the gate during
            training to the desired number of non-zeros.
          z_initializer: An optional initializer for z_logits.
          w_initializer: An optional initializer for w_logits.
        """
        super(DSelectKGate, self).__init__()
        self.task = task
        self._num_nonzeros = num_nonzeros
        self._smooth_step = SmoothStep(gamma)
        self._entropy_reg = entropy_reg
        self._z_initializer = z_initializer or tf.keras.initializers.RandomUniform(
            -gamma / 100, gamma / 100)
        self._w_initializer = w_initializer or tf.keras.initializers.RandomUniform()
        self.zeta = zeta
        self.load_balancing_penalty = load_balancing_penalty
        self.exp_decay_mov_ave = exp_decay_mov_ave

    def build(self, input_shape):
        """Creates the layer's internal variables."""
        if isinstance(input_shape, tuple):
            expert_shapes, routing_input_shape, permutation_shape = input_shape
        else:
            expert_shapes, routing_input_shape, _ = input_shape, None
        num_experts = len(expert_shapes)
        self.num_experts = num_experts
        # num_binary is the number of binary vars required to encode the
        # num_experts suggest_categoricals.
        self._num_binary = math.ceil(math.log2(num_experts))
        # Boolean to check if num_experts is a power of 2.
        self._power_of_2 = (num_experts == 2**self._num_binary)
        if routing_input_shape is None:
            # z_logits is a trainable 3D tensor used for selecting the experts.
            # Axis 0: Number of non-zero experts to select.
            # Axis 1: Dummy axis of length 1 used for broadcasting.
            # Axis 2: Each num_binary-dimensional row corresponds to a "single-expert"
            # selector.
            self._z_logits = self.add_weight(
                name="z_logits",
                shape=(self._num_nonzeros, 1, self._num_binary),
                initializer=self._z_initializer,
                trainable=True)
            # w_logits is a trainable tensor used to assign weights to the
            # single-expert selectors. Each element of w_logits is a logit.
            self._w_logits = self.add_weight(
                name="w_logits",
                shape=(self._num_nonzeros, 1),
                initializer=self._w_initializer,
                trainable=True)
        else:
            self._z_logits = tf.keras.layers.Dense(
                self._num_nonzeros * self._num_binary,
                kernel_initializer=self._z_initializer,
                bias_initializer=self._z_initializer)
            self._w_logits = tf.keras.layers.Dense(
                self._num_nonzeros,
                kernel_initializer=self._w_initializer,
                bias_initializer=self._w_initializer)
            
        # binary_matrix is a (num_experts, num_binary)-matrix used for binary
        # encoding. The i-th row contains a num_binary-digit binary encoding of the
        # integer i.
        binary_matrix = np.array([
            list(np.binary_repr(val, width=self._num_binary))
            for val in range(num_experts)
        ]).astype(bool)
        # A constant tensor = binary_matrix, with an additional dimension for
        # broadcasting.
        self._binary_codes = tf.expand_dims(
            tf.constant(binary_matrix, dtype=bool), axis=0)
        self.built = True

    def call(self, inputs, training=False):
        """Calls the layer.
        Args:
          inputs: For task-only conditioning, the input should be a list of tensors,
          each corresponding to an expert. For example conditioning, the input
          should be a tuple of the form: (experts, routing_inputs), where experts is
          a list of expert tensors, and routing_inputs is a 2D tensor of input
          examples.
          training: True if the call is in training mode.
        Returns:
          A (single) tensor representing the mixture of experts.
        """
        if isinstance(inputs, tuple):
            experts, routing_inputs, permutation_weights = inputs
        else:
            experts, routing_inputs = inputs, None
        if routing_inputs is not None:
            # Example-conditioned routing.
            expert_weights, selector_outputs, selector_outputs_before_perm  = (
                self._compute_example_conditioned_expert_weights(routing_inputs, permutation_weights)
            )

#             # Add load balancing loss
#             load_balancing_loss = control_flow_util.smart_cond(
#                 training,
#                 lambda: tf.cond(
#                     tf.math.equal(
#                         tf.cast(self.load_balancing_penalty, self.dtype),
#                         tf.constant(0.0, dtype=self.dtype)
#                     ),
#                     lambda: tf.zeros(()),
#                     lambda: self._compute_load_balancing_loss(expert_weights)
#                 ),
#                 lambda: tf.zeros(())
#             )
#             self.add_loss(self.load_balancing_penalty*load_balancing_loss)

            # if not training:
              # tf.print(expert_weights[10:15,:], summarize=-1, output_stream=sys.stdout)
            s_concat = tf.where(
                tf.math.less(expert_weights,1e-5),
                tf.ones_like(expert_weights),
                tf.zeros_like(expert_weights)
            )

            self.add_metric(
                tf.reduce_mean(s_concat),
                name='avg_sparsity'
            )
            soft_averages = tf.reduce_mean(expert_weights, axis=[0]) # (num_experts,)
            hard_averages = tf.reduce_mean(tf.ones_like(s_concat)-s_concat, axis=[0]) # (num_experts,)
            soft_averages_for_all_experts_list = tf.split(
                tf.reshape(soft_averages, [-1]),
                self.num_experts
            )
            [self.add_metric(le, name='soft_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(soft_averages_for_all_experts_list)]
            hard_averages_for_all_experts_list = tf.split(
                tf.reshape(hard_averages, [-1]),
                self.num_experts
            )
            [self.add_metric(le, name='hard_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(hard_averages_for_all_experts_list)]

            simplex_constraint = tf.reduce_mean(
                tf.reduce_sum(expert_weights, axis=-1),
            )
            self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))

            simplex_constraint_fails = tf.reduce_sum(expert_weights, axis=-1) # (b, )

            simplex_constraint_fails = tf.where(
                tf.math.less(simplex_constraint_fails, 1.0-1e-5),
                tf.ones_like(simplex_constraint_fails),
                tf.zeros_like(simplex_constraint_fails)
            ) # (b, )
            simplex_constraint_fails = tf.reduce_mean(simplex_constraint_fails, axis=0)
            self.add_metric(simplex_constraint_fails, name='simplex_constraint_fails_for_task_{}'.format(self.task+1))

            output = tf.math.accumulate_n([
                tf.reshape(expert_weights[:, i], [-1, 1]) * experts[i]
                for i in range(len(experts))
            ])
        else:
            # Task-only routing.
            expert_weights, selector_outputs = self._compute_expert_weights()
            self.add_metric(
                tf.reduce_mean(
                    tf.where(
                        tf.math.less(expert_weights,1e-5),
                        tf.ones_like(expert_weights),
                        tf.zeros_like(expert_weights)
                    )
                ),
                name='avg_sparsity'
            )
            output = tf.math.accumulate_n(
                [expert_weights[i] * experts[i] for i in range(len(experts))])
        if training:
            self._add_regularization_loss(selector_outputs_before_perm)

        return output, soft_averages, hard_averages

    def _compute_expert_weights(self):
        """Computes the weight vector for the experts.
        Args: None.
        Returns:
          A tuple: (expert_weights, selector_outputs).
            expert_weights is the final weight vector of the experts.
            selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
            represents the outputs of the i-th single-expert selector.
        """
        # Shape = (num_nonzero, 1, num_binary).
        smooth_step_activations = self._smooth_step(self._z_logits)
        # Shape = (num_nonzero, num_experts).
        selector_outputs = tf.math.reduce_prod(
            tf.where(self._binary_codes, smooth_step_activations,
                     1 - smooth_step_activations),
            axis=2)
        # Weights for the single-expert selectors: shape = (num_nonzero, 1).
        selector_weights = tf.nn.softmax(self._w_logits, axis=0)
        expert_weights = tf.math.reduce_sum(
            selector_weights * selector_outputs, axis=0)

        return expert_weights, selector_outputs

    def _compute_example_conditioned_expert_weights(self, routing_inputs, permutation_weights):
        """Computes the example-conditioned weights for the experts.
        Args:
          routing_inputs: a tensor of shape=(batch_size, num_features) containing
            the input examples.
        Returns:
          A tuple: (expert_weights, selector_outputs).
            expert_weights is a tensor with shape=(batch_size, num_experts),
            containing the expert weights for each example in routing_inputs.
            selector_outputs is a tensor with
            shape=(batch_size, num_nonzero, num_experts), which contains the outputs
            of the single-expert selectors for all the examples in routing_inputs.
        """
        sample_logits = tf.reshape(
            self._z_logits(routing_inputs),
            [-1, self._num_nonzeros, 1, self._num_binary])
        smooth_step_activations = self._smooth_step(sample_logits)

        # Shape = (batch_size, num_nonzeros, num_experts).
        selector_outputs = tf.math.reduce_prod(
            tf.where(
                tf.expand_dims(self._binary_codes, 0), smooth_step_activations,
                1 - smooth_step_activations), 3)

        # selector_outputs: (batch_size, num_nonzeros, num_experts), permutation_weights: [num_nonzeros, num_experts, num_experts]
        selector_outputs_permuted = tf.einsum('bjk,jkl->bjl', selector_outputs, permutation_weights) # (batch_size, num_nonzeros, num_experts)
#         selector_outputs_permuted = selector_outputs_permuted/tf.reduce_sum(selector_outputs_permuted, axis=-1, keepdims=True)  # (batch_size, num_nonzeros, num_experts)
#         selector_outputs_permuted = selector_outputs

        # Weights for the single-expert selectors.
        # Shape = (batch_size, num_nonzeros, 1).
        selector_weights = tf.expand_dims(self._w_logits(routing_inputs), 2)
        selector_weights = tf.nn.softmax(selector_weights, axis=1)

        # Sum over the single-expert selectors. Shape = (batch_size, num_experts).
        expert_weights = tf.math.reduce_sum(
            selector_weights * selector_outputs_permuted, axis=1)

        return expert_weights, selector_outputs_permuted, selector_outputs

    def _add_regularization_loss(self, selector_outputs):
        """Adds regularization loss based on the selector outputs.
        Args:
            selector_outputs: a tensor with shape (batch_size, num_nonzero,
                num_experts) or (num_nonzero, num_experts), where the last dimension
                stores the weight vector of each single-expert selector.
        """
        if self._entropy_reg is not None:
            # Add entropy regularization to each single-expert selector to encourage sparsity.
            self.add_loss(self._entropy_reg(selector_outputs))

        if not self._power_of_2:
            # If the number of experts is not a power of 2, we add a regularization
            # term to prevent the "non-reachable" experts from getting all the nonzero
            # weights for any single-expert selector. The regularization term is equal
            # to 1/sum(weights of reachable experts) so that the reachable experts
            # cannot get zero weights.
            # In case of example conditioning, this regularizer is added per example.
            # NOTE: This regularization term has no effect once the sum of the weights
            # of the reachable experts reaches 1, which is the typical/expected case.
            # tf.print('========selector_outputs', tf.math.reduce_mean(selector_outputs,axis=0))
            recip_loss = tf.math.reduce_sum(1 / (1e-6 + tf.math.reduce_sum(selector_outputs, axis=-1)))
            # tf.print('========loss', recip_loss)
            self.add_loss(self.zeta*recip_loss)

#     def _compute_load_balancing_loss(self, weights, eps=1e-10, symmetric=False):
#         """ Compute KL divergence between uniform distribution and weight distribution

#         KLD1 = np.sum(P*np.log(P*n+1e-10))
#         KLD2 = -np.sum(np.log(P*n+1e-10))/n
#         KLD = 0.5*(KLD1+KLD2)

#         Args:
#             weights: # (b, num_experts)
#         """
#         weights_ave_current = tf.reduce_mean(weights, axis=[0]) # (num_experts, )
#         weights_ave_current = weights_ave_current/(tf.reduce_sum(weights_ave_current, axis=-1, keepdims=True)+1e-6)  # (num_experts,)
        
#         weights_ave = (1-self.exp_decay_mov_ave)*tf.cast(weights_ave_current, dtype=self.dtype) + self.exp_decay_mov_ave*tf.cast(self.weights_ave_past, dtype=self.dtype)
#         self.weights_ave_past.assign(weights_ave)

#         kld = tf.reduce_sum(
#             weights_ave * tf.math.log(weights_ave*self.num_experts + eps)
#         )
#         if symmetric:
#             kld2 = tf.reduce_sum(
#                 -tf.math.log(weights_ave*self.num_experts + eps)
#             ) / tf.cast(self.num_experts, dtype=self.dtype)
#             kld = 0.5*(kld+kld2)
#         tf.print("===========kld:", kld)
#         self.add_metric(kld, name="load_balancing_loss")
#         return kld

#   public version of the private method to compute expert weights
#   @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
#   def compute_expert_weights(self,_=0.):  # dummy input, since tf.function need at least an arg for signature
#     """Computes the weight vector for the experts. Publicly accessible version of the above method.
#     Args: None.
#     Returns:
#       A tuple: (expert_weights, selector_outputs).
#         expert_weights is the final weight vector of the experts.
#         selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
#         represents the outputs of the i-th single-expert selector.
#     """
#     # Shape = (num_nonzero, 1, num_binary).
#     smooth_step_activations = self._smooth_step(self._z_logits)
#     # Shape = (num_nonzero, num_experts).
#     selector_outputs = tf.math.reduce_prod(
#         tf.where(self._binary_codes, smooth_step_activations,
#                  1 - smooth_step_activations),
#         axis=2)
#     # Weights for the single-expert selectors: shape = (num_nonzero, 1).
#     selector_weights = tf.nn.softmax(self._w_logits, axis=0)
#     expert_weights = tf.math.reduce_sum(
#         selector_weights * selector_outputs, axis=0)

#     return expert_weights, selector_outputs

class DSelectKGateOld(tf.keras.layers.Layer):
  """A custom layer for selecting a sparse mixture of experts.
  Let f_1, f_2, ..., f_n be the experts. The layer returns:
              a_1 * f_1 + a_2 * f_2 + ... + a_n * f_n,
  where the mixture weights satisfy a_1, ..., a_n >= 0 and a_1 + ... + a_n = 1.
  The number of non-zeros in the mixture weights can be directly controlled.
  The layer is differentiable and can be trained using first-order methods like
  SGD.
  Input: For task-only conditioning, the input should be a list of tensors,
    each corresponding to an expert. For example conditioning, the input should
    be a tuple of the form: (experts, routing_inputs), where experts is a list
    of expert tensors, and routing_inputs is a 2D tensor of input examples.
    Note: In both cases, the expert tensors should have the same shape.
  Output: Tensor, with the same shape as the expert tensors.
  Example:
    # Construct a DSelectKGate to select 2 out of 4 experts.
    gate = DSelectKGate(num_nonzeros=2)
    # output_tensor is a sparse mixture of the 4 tensors in the inputs.
    output_tensor = gate(inputs)
  """

  def __init__(
      self,
      num_nonzeros,
      gamma = 1.0,
      entropy_reg = None,
      z_initializer = None,
      w_initializer = None,
      zeta = 1.0,
      task=0,
  ):
    """DSelectKGate constructor.
    Args:
      num_nonzeros: An upper bound on the number of non-zeros in the mixture
        weights (a_1, ..., a_n), where n is the number of experts inferred from
        the inputs to this layer.
      gamma: A scaling parameter for the smooth-step function.
      entropy_reg: An optional `EntropyRegularizer`. This regularizer, if set,
        can be used to control the speed of convergence of the gate during
        training to the desired number of non-zeros.
      z_initializer: An optional initializer for z_logits.
      w_initializer: An optional initializer for w_logits.
    """
    super(DSelectKGate, self).__init__()
    self.task = task
    self._num_nonzeros = num_nonzeros
    self._smooth_step = SmoothStep(gamma)
    self._entropy_reg = entropy_reg
    self._z_initializer = z_initializer or tf.keras.initializers.RandomUniform(
        -gamma / 100, gamma / 100)
    self._w_initializer = w_initializer or tf.keras.initializers.RandomUniform()
    self.zeta = zeta

  def build(
      self, input_shape
  ):
    """Creates the layer's internal variables."""
    if isinstance(input_shape, tuple):
      expert_shapes, routing_input_shape, _ = input_shape
    else:
      expert_shapes, routing_input_shape, _ = input_shape, None
    num_experts = len(expert_shapes)
    self.nb_experts = num_experts
    # num_binary is the number of binary vars required to encode the
    # num_experts suggest_categoricals.
    self._num_binary = math.ceil(math.log2(num_experts))
    # Boolean to check if num_experts is a power of 2.
    self._power_of_2 = (num_experts == 2**self._num_binary)
    if routing_input_shape is None:
      # z_logits is a trainable 3D tensor used for selecting the experts.
      # Axis 0: Number of non-zero experts to select.
      # Axis 1: Dummy axis of length 1 used for broadcasting.
      # Axis 2: Each num_binary-dimensional row corresponds to a "single-expert"
      # selector.
      self._z_logits = self.add_weight(
          name="z_logits",
          shape=(self._num_nonzeros, 1, self._num_binary),
          initializer=self._z_initializer,
          trainable=True)
      # w_logits is a trainable tensor used to assign weights to the
      # single-expert selectors. Each element of w_logits is a logit.
      self._w_logits = self.add_weight(
          name="w_logits",
          shape=(self._num_nonzeros, 1),
          initializer=self._w_initializer,
          trainable=True)
    else:
      self._z_logits = tf.keras.layers.Dense(
          self._num_nonzeros * self._num_binary,
          kernel_initializer=self._z_initializer,
          bias_initializer=self._z_initializer)
      self._w_logits = tf.keras.layers.Dense(
          self._num_nonzeros,
          kernel_initializer=self._w_initializer,
          bias_initializer=self._w_initializer)
    # binary_matrix is a (num_experts, num_binary)-matrix used for binary
    # encoding. The i-th row contains a num_binary-digit binary encoding of the
    # integer i.
    binary_matrix = np.array([
        list(np.binary_repr(val, width=self._num_binary))
        for val in range(num_experts)
    ]).astype(bool)
    # A constant tensor = binary_matrix, with an additional dimension for
    # broadcasting.
    self._binary_codes = tf.expand_dims(
        tf.constant(binary_matrix, dtype=bool), axis=0)
    self.built = True

  def call(self,
           inputs,
           training = False):
    """Calls the layer.
    Args:
      inputs: For task-only conditioning, the input should be a list of tensors,
      each corresponding to an expert. For example conditioning, the input
      should be a tuple of the form: (experts, routing_inputs), where experts is
      a list of expert tensors, and routing_inputs is a 2D tensor of input
      examples.
      training: True if the call is in training mode.
    Returns:
      A (single) tensor representing the mixture of experts.
    """
    if isinstance(inputs, tuple):
      experts, routing_inputs, _ = inputs
    else:
      experts, routing_inputs = inputs, None
    if routing_inputs is not None:
      # Example-conditioned routing.
      expert_weights, selector_outputs = (
          self._compute_example_conditioned_expert_weights(routing_inputs)
      )
      # if not training:
          # tf.print(expert_weights[10:15,:], summarize=-1, output_stream=sys.stdout)
      s_concat = tf.where(
        tf.math.less(expert_weights,1e-5),
        tf.ones_like(expert_weights),
        tf.zeros_like(expert_weights)
      )

      self.add_metric(
          tf.reduce_mean(s_concat),
          name='avg_sparsity'
      )
      soft_averages = tf.reduce_mean(expert_weights, axis=[0]) # (nb_experts,)
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
        tf.reduce_sum(expert_weights, axis=-1),
      )
      self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))
    
      simplex_constraint_fails = tf.reduce_sum(expert_weights, axis=-1) # (b, )

      simplex_constraint_fails = tf.where(
        tf.math.less(simplex_constraint_fails, 1.0-1e-5),
        tf.ones_like(simplex_constraint_fails),
        tf.zeros_like(simplex_constraint_fails)
      ) # (b, )
      simplex_constraint_fails = tf.reduce_mean(simplex_constraint_fails, axis=0)
      self.add_metric(simplex_constraint_fails, name='simplex_constraint_fails_for_task_{}'.format(self.task+1))
    
      output = tf.math.accumulate_n([
          tf.reshape(expert_weights[:, i], [-1, 1]) * experts[i]
          for i in range(len(experts))
      ])
    else:
      # Task-only routing.
      expert_weights, selector_outputs = self._compute_expert_weights()
      self.add_metric(
          tf.reduce_mean(
              tf.where(
                  tf.math.less(expert_weights,1e-5),
                  tf.ones_like(expert_weights),
                  tf.zeros_like(expert_weights)
              )
          ),
          name='avg_sparsity'
      )
      output = tf.math.accumulate_n(
          [expert_weights[i] * experts[i] for i in range(len(experts))])
    if training:
      self._add_regularization_loss(selector_outputs)

    return output

  def _compute_expert_weights(self):
    """Computes the weight vector for the experts.
    Args: None.
    Returns:
      A tuple: (expert_weights, selector_outputs).
        expert_weights is the final weight vector of the experts.
        selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
        represents the outputs of the i-th single-expert selector.
    """
    # Shape = (num_nonzero, 1, num_binary).
    smooth_step_activations = self._smooth_step(self._z_logits)
    # Shape = (num_nonzero, num_experts).
    selector_outputs = tf.math.reduce_prod(
        tf.where(self._binary_codes, smooth_step_activations,
                 1 - smooth_step_activations),
        axis=2)
    # Weights for the single-expert selectors: shape = (num_nonzero, 1).
    selector_weights = tf.nn.softmax(self._w_logits, axis=0)
    expert_weights = tf.math.reduce_sum(
        selector_weights * selector_outputs, axis=0)

    return expert_weights, selector_outputs

  def _compute_example_conditioned_expert_weights(self, routing_inputs):
    """Computes the example-conditioned weights for the experts.
    Args:
      routing_inputs: a tensor of shape=(batch_size, num_features) containing
        the input examples.
    Returns:
      A tuple: (expert_weights, selector_outputs).
        expert_weights is a tensor with shape=(batch_size, num_experts),
        containing the expert weights for each example in routing_inputs.
        selector_outputs is a tensor with
        shape=(batch_size, num_nonzero, num_experts), which contains the outputs
        of the single-expert selectors for all the examples in routing_inputs.
    """
    sample_logits = tf.reshape(
        self._z_logits(routing_inputs),
        [-1, self._num_nonzeros, 1, self._num_binary])
    smooth_step_activations = self._smooth_step(sample_logits)
    # Shape = (batch_size, num_nonzeros, num_experts).
    selector_outputs = tf.math.reduce_prod(
        tf.where(
            tf.expand_dims(self._binary_codes, 0), smooth_step_activations,
            1 - smooth_step_activations), 3)
    # Weights for the single-expert selectors.
    # Shape = (batch_size, num_nonzeros, 1).
    selector_weights = tf.expand_dims(self._w_logits(routing_inputs), 2)
    selector_weights = tf.nn.softmax(selector_weights, axis=1)
    # Sum over the single-expert selectors. Shape = (batch_size, num_experts).
    expert_weights = tf.math.reduce_sum(
        selector_weights * selector_outputs, axis=1)

    return expert_weights, selector_outputs

  def _add_regularization_loss(self, selector_outputs):
    """Adds regularization loss based on the selector outputs.
    Args:
      selector_outputs: a tensor with shape (batch_size, num_nonzero,
        num_experts) or (num_nonzero, num_experts), where the last dimension
        stores the weight vector of each single-expert selector.
    """
    if self._entropy_reg is not None:
      # Add entropy regularization to each single-expert selector to encourage
      # sparsity.
      self.add_loss(self._entropy_reg(selector_outputs))

    if not self._power_of_2:
      # If the number of experts is not a power of 2, we add a regularization
      # term to prevent the "non-reachable" experts from getting all the nonzero
      # weights for any single-expert selector. The regularization term is equal
      # to 1/sum(weights of reachable experts) so that the reachable experts
      # cannot get zero weights.
      # In case of example conditioning, this regularizer is added per example.
      # NOTE: This regularization term has no effect once the sum of the weights
      # of the reachable experts reaches 1, which is the typical/expected case.
      # tf.print('========selector_outputs', tf.math.reduce_mean(selector_outputs,axis=0))
      recip_loss = tf.math.reduce_sum(1 / (1e-6 + tf.math.reduce_sum(selector_outputs, axis=-1)))
      # tf.print('========loss', recip_loss)
      self.add_loss(self.zeta*recip_loss)

  # public version of the private method to compute expert weights
#   @tf.function(input_signature=[tf.TensorSpec(shape=[],dtype=tf.float32)])
#   def compute_expert_weights(self,_=0.):  # dummy input, since tf.function need at least an arg for signature
#     """Computes the weight vector for the experts. Publicly accessible version of the above method.
#     Args: None.
#     Returns:
#       A tuple: (expert_weights, selector_outputs).
#         expert_weights is the final weight vector of the experts.
#         selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
#         represents the outputs of the i-th single-expert selector.
#     """
#     # Shape = (num_nonzero, 1, num_binary).
#     smooth_step_activations = self._smooth_step(self._z_logits)
#     # Shape = (num_nonzero, num_experts).
#     selector_outputs = tf.math.reduce_prod(
#         tf.where(self._binary_codes, smooth_step_activations,
#                  1 - smooth_step_activations),
#         axis=2)
#     # Weights for the single-expert selectors: shape = (num_nonzero, 1).
#     selector_weights = tf.nn.softmax(self._w_logits, axis=0)
#     expert_weights = tf.math.reduce_sum(
#         selector_weights * selector_outputs, axis=0)

#     return expert_weights, selector_outputs





if __name__ == "__main__":
  config = {
    "num_nonzeros": 3,
    "input_size": 128,
    "gamma": 1.0,
    "use_routing_input": True,
    "use_in_training": True,
    "entropy_reg": 5.0,
    "z_initializer": None,
    "w_initializer": None,
    "zeta": 1.0,
    "nb_experts": 4
  }

  # gate = DSelectKWrapperGate(
  #   config["num_nonzeros"],
  #   config["gamma"],
  #   config["entropy_reg"],
  #   config["z_initializer"],
  #   config["w_initializer"]
  # )
  gate = DSelectKWrapperGate(config)

  # # model = tf.keras.Sequential(layers=[gate])
  model = gate
  h = [
      np.random.random((8, 128)) for _ in range(config["nb_experts"])
  ]
  x = np.random.random((8, 64))
  input_shape = ([(None,128) for _ in range(6)], (None, 64))
  # model.build(input_shape)
  model((h,x), training=True)





  # weights_gate1 = [np.array([[[-4.8350784e-01, -3.8189444e-01, -3.1637534e-04]],
  #  [[-4.7250253e-01, -4.0204462e-01, -5.5637183e-03]],
  #  [[-4.3926120e-01, -4.4253212e-01, -9.5844280e-04]]], dtype=np.float32), np.array([[-0.04460425],
  #  [-0.03426867],
  #  [-0.00679468]], dtype=np.float32)]

  # weights_gate2 = [np.array([[[-0.48681504, -0.37276113, 0.00889401]],
  #  [[-0.3775608 , -0.48478568,  0.00527356]],
  #  [[-0.37430385, -0.48626095,  0.00303185]]], dtype=np.float32), np.array([[-0.01771483],
  #  [ 0.02306351],
  #  [-0.04829251]], dtype=np.float32)]

  # weights_gate3 = [np.array([[[-0.4820841 , -0.38301602, -0.00407531]],
  #  [[-0.44588712, -0.43429336,  0.00831761]],
  #  [[-0.38366124, -0.48167253,  0.00457889]]], dtype=np.float32), np.array([[ 0.01693494],
  #  [ 0.01659504],
  #  [-0.04026721]], dtype=np.float32)]

  # weights_gate4 = [np.array([[[-0.41485968, -0.4634473 , -0.01299429]],
  #  [[-0.49260667, -0.35545436,  0.00512811]],
  #  [[-0.46284005, -0.41576687, -0.00715848]]], dtype=np.float32), np.array([[-0.00175389],
  #  [ 0.0309787 ],
  #  [-0.04728409]], dtype=np.float32)]

  # model.set_weights(weights_gate4)

  # print(model._num_nonzeros)

  # print(model._compute_expert_weights()[0])


  # config = {
  #   "num_nonzeros": 4,
  #   "input_size": 128,
  #   "gamma": 10,
  #   "use_routing_input": False,
  #   "use_in_training": True,
  #   "entropy_reg": None,
  #   "z_initializer": None,
  #   "w_initializer": None
  # }
  # weights_gate = [np.array([[[ 0.03187786, -0.07982165,  0.06884833]],
  #      [[-0.08544275,  0.08648551,  0.01020409]],
  #      [[-0.02168947, -0.01692589, -0.01999864]],
  #      [[ 0.02248277,  0.00930842, -0.00441247]]], dtype=np.float32), np.array([[ 0.01316368],
  #      [-0.04914904],
  #      [-0.04939564],
  #      [ 0.00583015]], dtype=np.float32)]

  # config = {
  #   "entropy_reg": 0.1,
  #   "gamma": 10,
  #   "input_size": 128,
  #   "num_nonzeros": 4,
  #   "use_in_training": True,
  #   "use_routing_input": False,
  #   "w_initializer": None,
  #   "z_initializer": None
  # }

  # gate = DSelectKGate(
  #   config["num_nonzeros"],
  #   config["gamma"],
  #   config["entropy_reg"],
  #   config["z_initializer"],
  #   config["w_initializer"]
  # )

  # weights_gate1 = [np.array([[[-4.9999976, -4.9999976,  4.9999976]],
  #      [[-4.9999976, -4.9999976,  4.9999976]],
  #      [[-4.9999976, -4.9999976, -4.9999976]],
  #      [[ 4.9999976, -4.9999976, -4.9999976]]], dtype=np.float32), np.array([[ 0.42471346],
  #      [ 0.4043398 ],
  #      [-0.2553278 ],
  #      [-0.51335585]], dtype=np.float32)]
  # weights_gate = [np.array([[[ 4.9999976,  4.9999976, -4.9999976]],
  #      [[-4.9999976, -4.9999976, -4.9999976]],
  #      [[-4.9999976, -4.9999976,  4.9999976]],
  #      [[-4.9999976,  4.9999976, -4.9999976]]], dtype=np.float32), np.array([[ 0.00120226],
  #      [ 0.06896388],
  #      [-0.6949055 ],
  #      [ 0.59256905]], dtype=np.float32)]

  # model = gate
  # input_shape = [(None,128) for _ in range(8)]
  # model.build(input_shape)

  # model.set_weights(weights_gate)

  # print(model._num_nonzeros)

  # print(model._compute_expert_weights()[0])

  


