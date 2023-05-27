"""
File containing utility functions to use in the main.py, train_tasks.py, and test_tasks.py files.
"""
import inspect
import tensorflow as tf
import tensorflow_addons as tfa
from munch import munchify      # we could have used easydict instead
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
import yaml
from data.dataset_openers import DatasetOpenerMapper


# We define here a custom LOSS object for MovieLens200K (to deal with NaN values).
class CustomMovielensMeanSquaredError(object):

    def __init__(
        self,
        forbidden_value=0
    ):
        super(CustomMovielensMeanSquaredError, self).__init__()
        self.underlying_loss = tf.keras.losses.MeanSquaredError()
        self.forbidden_value = forbidden_value

    def __call__(
        self,
        y_true,
        y_pred
    ):
        indices_to_keep = (y_true != self.forbidden_value)
        y_true = y_true[indices_to_keep]
        y_pred = y_pred[indices_to_keep]
        return self.underlying_loss(
            y_true,
            y_pred
        )

# We define here a custom METRIC object for MovieLens200K (to deal with NaN values).
class MaskedMovielensMeanSquaredError(object):
    
    def __init__(
        self,
        forbidden_value=0
    ):
        super(MaskedMovielensMeanSquaredError, self).__init__()
        self.underlying_metric = tf.keras.metrics.MeanSquaredError()
        self.forbidden_value = forbidden_value

    def update_state(
        self,
        y_true,
        y_pred
    ):
        indices_to_keep = (y_true != self.forbidden_value)
        y_true = y_true[indices_to_keep]
        y_pred = y_pred[indices_to_keep]
        self.underlying_metric.update_state(y_true, y_pred)

    def result(
        self
    ):
        return self.underlying_metric.result()    

    def reset_state(
        self
    ):
        try:
            self.underlying_metric.reset_state()
        except:
            self.underlying_metric.reset_states()        

class CustomMovielensRootMeanSquaredError(object):
    
    def __init__(
        self,
        forbidden_value=0
    ):
        super(CustomMovielensRootMeanSquaredError, self).__init__()
        self.underlying_metric = tf.keras.metrics.RootMeanSquaredError()
        self.forbidden_value = forbidden_value

    def update_state(
        self,
        y_true,
        y_pred
    ):
        indices_to_keep = (y_true != self.forbidden_value)
        y_true = y_true[indices_to_keep]
        y_pred = y_pred[indices_to_keep]
        self.underlying_metric.update_state(y_true, y_pred)

    def result(
        self
    ):
        return self.underlying_metric.result()    

    def reset_state(
        self
    ):
        try:
            self.underlying_metric.reset_state()
        except:
            self.underlying_metric.reset_states()        

            
# We define a loss mapper function; maps a loss object name from the task config to an actual loss class from TF.
def LossMapper(type):
    if type=="CustomMovielensMeanSquaredError": 
        return CustomMovielensMeanSquaredError
    return getattr(tf.keras.losses, type)


# We define an optimizer mapper function; 
# maps an optimizer object name from the train config to an actual optimizer class from TF.
def OptimizerMapper(type):
    try: 
        optimizer = getattr(tf.keras.optimizers, type)
    except:
        optimizer = getattr(tfa.optimizers, type)
    return optimizer


# We define a metric mapper function; 
# maps a metric object name from the task config to an actual metric class from TF.
def MetricMapper(type):
    if type=="CustomMovielensRootMeanSquaredError": 
        return CustomMovielensRootMeanSquaredError
    elif type=="MaskedMovielensMeanSquaredError": 
        return MaskedMovielensMeanSquaredError
    try:
        metric = getattr(tf.keras.metrics, type)
    except:
        metric = getattr(tfa.metrics, type)
    return metric


# We define a callback mapper function; 
# maps a callback object name from the train config to an actual callback class from TF.
def CallbackMapper(type):
    return getattr(tf.keras.callbacks, type)


def _convert_params_recursive(config_dictionary, trial=None):
    """
    Parses a config Python dictionary to instantiate the right optuna objects when mentioned.
    If no optuna object is mentioned, then the function does not do anything.
    inputs: a Python dict potentially containing the "optuna_type" key
    outputs: None (inplace modification).
    """
    if trial is not None:
        def OptunaMapper(type):
            return getattr(trial, type)
    if isinstance(config_dictionary, list) and len(config_dictionary) > 0 and isinstance(config_dictionary[0],dict):
        for obj in config_dictionary:
            _convert_params_recursive(obj, trial=trial)
    if isinstance(config_dictionary, dict):
        for key in config_dictionary:
            if isinstance(config_dictionary[key],dict) and "optuna_type" in config_dictionary[key].keys():
                if config_dictionary[key]["optuna_type"] != "suggest_categorical":
                    config_dictionary[key] = OptunaMapper(
                        config_dictionary[key]["optuna_type"]
                    )(
                        key, 
                        config_dictionary[key]["values"][0], 
                        config_dictionary[key]["values"][1]
                    )
                elif config_dictionary[key]["optuna_type"] == "suggest_categorical":
                    config_dictionary[key] = OptunaMapper(
                        config_dictionary[key]["optuna_type"]
                    )(
                        key, 
                        config_dictionary[key]["values"]
                    )
                else:
                    config_dictionary[key] = OptunaMapper(
                        config_dictionary[key]["optuna_type"]
                    )(
                        key, 
                        config_dictionary[key]["values"]
                    )                    
            else:
                _convert_params_recursive(config_dictionary[key], trial=trial)
    return


def MergeConfigs(model_config, train_config, trial=None):
    """
    Simply parses and merges the model, training, and task configurations into a single Munch object.
    inputs: 2 dictionaries:
        - model config dict (model architecture mentioned)
        - train config dict (training hyperparameters mentioned)
        - trial object from optuna
    outputs: a Munch object with all the keys from the 2 dictionaries (and optuna functions too)
    """
    _convert_params_recursive(model_config, trial=trial)
    _convert_params_recursive(train_config, trial=trial)
    general_config = {
        **model_config,
        **train_config,
    }
    for gate in general_config["gates"]:
        if gate["module"] == "TrimmedLassoSimplexProjGate":
            gate["params"]["gate_learning_rate"] = general_config["optimizer"]["params"]["learning_rate"]
    return munchify(general_config)


def LoadDatasets(
    config, 
#     trainval=True
):
    """
    Depending on the taskset_ids selected in the train_config file and the task_configs, 
    this function loads the datasets linked to the taskset_ids selected, 
    using the dataset openers obtained with the mapper in /data/dataset_openers.
    inputs:
        - merged config
        - trainval (bool): whether to return train and val datasets or just test dataset
    outputs:
        - train and val datasets
        - OR test dataset only
    """ 
    # We import the shared X, and each specific y IN THE SAME ORDER as specified in the task_config file.
    taskset_name = config["taskset"]["name"]
    dataset_opener_function = DatasetOpenerMapper[
        taskset_name
    ]
    # the task_names list will help us import the targets associated to each subtask in the dataset
    task_names = [task["name"] for task in config["taskset"]["tasks"]]
    # TODO: simplify this
    train_dataset, val_dataset, test_dataset = dataset_opener_function(
        config["taskset"]["raw_data_bpath"],
        task_names,
        taskset_name,
        config["num_experts"]
#             config["train_batch_size"],
#             config["val_batch_size"]
    )
    return train_dataset, val_dataset, test_dataset
#     else:
#         test_dataset = dataset_opener_function(
#             config["taskset"]["raw_data_bpath"],
#             task_names,
#             taskset_name,
# #             config["train_batch_size"],
# #             config["val_batch_size"],
#             trainval=trainval
#         )
#         return test_dataset        


def LoadLosses(config):
    """
    Depending on the taskset_ids selected in the train_config file and the task_configs,
    this function loads the losses linked to the taskset_ids selected,
    using the loss mapper just above.
    inputs: 
        - merged config object
    outputs: 
        - a list of the form [{"loss": Loss1, "weight": 0.2}], with the loss object Loss1 already instantiated, 
        and this for each task in the same order as in the "tasks" key from config.
        (the full convex combination loss cannot be instanciated here once and for all 
        since we cannot add functions when they are not evaluated)
    """
    task_losses_list = []
    for i,task_object in enumerate(config["taskset"]["tasks"]):
        task_losses_list.append({
            "loss": (
                LossMapper(task_object["loss"])() 
            ),
            "weight": config["task_weights"][i]
        })
    return task_losses_list


def LoadMetrics(config):
    """
    Depending on the task_ids selected in the train_config file and the task_configs,
    this function loads the metrics linked to the task_ids selected,
    using the metric mapper just above.
    inputs: 
        - merged config object
    outputs: 
        - a list of the form [{"metric1": Metric1, "metric2": Metric2}], with the loss object Loss1 already instantiated
    """
    task_metrics_list = []
    for task_object in config["taskset"]['tasks']:
        task_metrics_list.append({
            metric: MetricMapper(metric)() for metric in task_object["metrics"]
        })        
    return task_metrics_list


def ForwardModel(
    model,
    batch,
    task_losses_list,
    task_metrics_list,
    training=True,
):
    """
    Performs a forward pass of the model.
    inputs:
        - a built tf.keras Model
        - a batch of the form (x_batch, y_batch)
        - a list of the task losses (already instantiated!) of the form [{"loss": Loss1, "weight": 0.2}] to use here 
        (IN THE SAME ORDER as the outputs of the model, ie the order of the heads and tasks in the model config file)
        - a list of the task metrics (already instantiated!) of the form [{"metric1": Metric1, "metric2": Metric2}] to use here 
        (IN THE SAME ORDER as the outputs of the model, ie the order of the heads and tasks in the model config file)
        - a boolean to specify if we should record gradients
    outputs: 
        - the predictions of the model for the batch
        - the total loss value on the combination of tasks
        - a list of dictionaries containing the metrics computed for each task
    """
    # WARNING: y_batch is a list of the targets for each task IN THE RIGHT ORDER 
    # (=> the global dataset should be built accordingly)
    x_batch, y_batch = batch
#     print(x_batch)
#     print(y_batch)
    # warning: preds is a list of tensors
    preds = model(x_batch['input'], training=training, indices=x_batch['indices'])
    
    batch_loss = sum([
        # WARNING: y_true, y_pred IN THIS ORDER
        loss_dict["weight"] * loss_dict["loss"](y_batch[list(y_batch.keys())[i]], preds[i])
        for i,loss_dict in enumerate(task_losses_list)
    ])
    batch_loss += sum(model.losses)

    for i,metrics_dict in enumerate(task_metrics_list):
        for metric in metrics_dict:
            # WARNING: y_true, y_pred IN THIS ORDER
            metrics_dict[metric].update_state(y_batch[list(y_batch.keys())[i]], preds[i])

    return preds, batch_loss


# In the following section, we define weight recovery evaluation functions;
# the first two are used for the actual matching of the learned VS true gate weights.
def compute_weight_assignment_cost(estimate, truth, metric="wdist",task_weights=None, tol=1e-4):
    """
    Computes weight assignment costs between our learned gate weights AND the gate weights from ground truth.
    inputs:
        - estimate: our learned gate weights
        - truth: the gate weights from ground truth
        - metric used to compute assignment costs
        - weights given to the different tasks simultaneously solved
        - tolerence for the exact matching of certain metrics (like accuracy)
    outputs:
        - matrix of metric costs for the assignment of estimate to truth.
    """
    assert metric in {"wdist","accuracy","FPR", "FNR", "FP", "FN"}
    assert estimate.shape == truth.shape
    n_task, n_exp = estimate.shape
    metric_mat = np.zeros((n_exp,n_exp))
    
    if task_weights is None:
        task_weights = np.ones(n_task)/n_task
    
    if metric == "wdist":
        for i in range(n_exp):
            for j in range(n_exp):
                metric_mat[i,j] = np.sum(np.abs(estimate[:,i]-truth[:,j])*task_weights)
    elif metric in {"accuracy","FPR","FNR"}:
        estimate =  np.where(estimate>tol, 1, 0)
        truth = np.where(truth>tol,1,0)
        if metric == "accuracy":
            for i in range(n_exp):
                for j in range(n_exp):
                    metric_mat[i,j] = np.sum(np.abs(estimate[:,i]-truth[:,j])*task_weights)
        elif metric == "FPR":
            task_weights = task_weights/np.sum(truth==0,axis=1)
            for i in range(n_exp):
                for j in range(n_exp):
                    metric_mat[i,j] = np.sum(np.abs(estimate[:,i]-truth[:,j])*(1-truth[:,j])*task_weights)
        elif metric == "FNR":
            task_weights = task_weights/np.sum(truth==1,axis=1)
            for i in range(n_exp):
                for j in range(n_exp):
                    metric_mat[i,j] = np.sum(np.abs(estimate[:,i]-truth[:,j])*(truth[:,j])*task_weights)
        elif metric == "FP":
            task_weights = task_weights/np.sum(truth==0,axis=1)
            for i in range(n_exp):
                for j in range(n_exp):
                    metric_mat[i,j] = np.sum(np.abs(estimate[:,i]-truth[:,j])*(1-truth[:,j])*task_weights)
        elif metric == "FN":
            task_weights = task_weights/np.sum(truth==1,axis=1)
            for i in range(n_exp):
                for j in range(n_exp):
                    metric_mat[i,j] = np.sum(np.abs(estimate[:,i]-truth[:,j])*(truth[:,j])*task_weights)
    return metric_mat


def AssignWeights(estimate, truth, metric="wdist",task_weights=None,tol=1e-4):
    """
    Matches weights from ground truth gate weights TO our learned gate weights 
    (using ILP, and based on the weight assignment costs computed with the previously defined function).
    inputs:
        - estimate: our learned gate weights
        - truth: the gate weights from ground truth
    outputs:
        - estimate_perm_indices: permutation of our learned weights to match them to the true weights
        - truth_perm_indices: permutation of the true weights to match them to the learned weights.
    """
    cost_mat = compute_weight_assignment_cost(estimate, truth, metric, task_weights, tol)
    return linear_sum_assignment(cost_mat)


# These last functions for the weight recovery are used to evaluate the quality of the weight matching previously done.
def _w_dist(estimate_vect, truth_vect):
    """
    Simply computes the L1 weight distance between the estimate_vect of gate weights learned and the truth_vect.
    inputs:
        - estimate: our learned gate weights
        - truth: the gate weights from ground truth
    outputs:
        - the L1 distance between the two input vectors.
    """
    return (1 / truth_vect.shape[0]) * np.sum(np.abs(estimate_vect - truth_vect))


def _compute_metrics_for_gate(estimate_vect_, truth_vect_, tol=1e-5):
    """
    AFTER MATCHING the indices of our two vectors of weights (learned VS ground truth) using ILP, this function computes
    distance metrics characterizing how good our assignment is.
    inputs:
        - estimate: our learned gate weights
        - truth: the gate weights from ground truth
    outputs:
        - the L1 distance between the two input vectors
        - the accuracy of the weight matching
        - the f1 score of the weight matching
        - the precision of the weight matching
        - the recall of the weight matching
    """
    estimate_vect = np.where(estimate_vect_>tol, 1, 0)
    truth_vect = np.where(truth_vect_>tol, 1, 0)
    
    accuracy = accuracy_score(truth_vect, estimate_vect)
    f1 = f1_score(truth_vect, estimate_vect)
    precision = precision_score(truth_vect, estimate_vect)
    recall = recall_score(truth_vect, estimate_vect)
    dist = _w_dist(estimate_vect_, truth_vect_)
    
    return dist, accuracy, f1, precision, recall 


# Last function for weight recovery
def WeightRecoveryEvaluation(estimate_vect_, truth_vect_, tol=1e-5, task_weights=None):
    """
    AFTER MATCHING the indices of our two vectors of weights (learned VS ground truth) using ILP, leverages the previous 
    function to log the weight recovery/weight matching evaluation metrics and return them.
    inputs:
        - estimate: our learned gate weights
        - truth: the gate weights from ground truth
        - tol for exact matching metrics (like accuracy)
        - weights given to the different tasks simultaneously solved
    outputs:
        - a dict with all relevant metrics characterizing the weight recovery
    """
    n_task, n_expert = truth_vect_.shape
    if task_weights is None:
        task_weights = np.ones(n_task)/n_task
        
    dists = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(n_task):
        dist, accuracy, f1, precision, recall = _compute_metrics_for_gate(
            estimate_vect_[i], 
            truth_vect_[i]
        )
        dists.append(dist)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(
            f"Metrics for task {i + 1}:"
        )
        print(
            f"\t - Weight distances: {round(dist,4)}"
        )
        print(
            f"\t - Accuracy: {round(accuracy,4)}"
        )
        print(
            f"\t - Precision: {round(precision,4)}"
        )
        print(
            f"\t - Recall: {round(recall,4)}"
        )
        print(
            f"\t - F1 score: {round(f1,4)} \n"
        )
        
    avg_dist = sum([score * task_weight for score, task_weight in zip(dists, task_weights)])
    avg_accuracy = sum([score * task_weight for score, task_weight in zip(accuracies, task_weights)])
    avg_precision = sum([score * task_weight for score, task_weight in zip(precisions, task_weights)])
    avg_recall = sum([score * task_weight for score, task_weight in zip(recalls, task_weights)])
    avg_f1 = sum([score * task_weight for score, task_weight in zip(f1s, task_weights)])

    print("\nMetrics averaged across tasks:")
    print(
        f"\t - Weight distances: {round(avg_dist,4)}"
    )
    print(
        f"\t - Accuracy: {round(avg_accuracy,4)}"
    )
    print(
        f"\t - Precision: {round(avg_precision,4)}"
    )
    print(
        f"\t - Recall: {round(avg_recall,4)}"
    )
    print(
        f"\t - F1 score: {round(avg_f1,4)} \n"
    )

    return {
        "dists": dists,
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "avg_dist": avg_dist,
        "avg_accuracy": avg_accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1
    }
    
  
# We define here a custom lr scheduler.
class LinearEpochGradualWarmupPolynomialDecayLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a Linear Epoch Gradual Warmup and Polynomial decay schedule.
    It is commonly observed that a linear ramp-up and monotonically decreasing learning rate, whose degree of change
    is carefully chosen, results in a better performing model. This schedule applies a polynomial decay function to an
    optimizer step, given a provided `low_learning_rate`, to reach an `peal_learning_rate` in the given `warmup_steps`,
    and reach a low_learning rate in the remaining steps via a polynomial decay.
    It requires a `step` value to compute the learning rate. You can just pass a TensorFlow variable that you
    increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning rate when passed the current optimizer step.
    This can be useful for changing the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        return ((low_learning_rate - peak_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer` as the learning rate.
    Example: Fit a model while ramping up from 0.01 to 0.1 in 1000 steps and decaying from 0.1 to 0.01 in 9000
        steps using sqrt (i.e. power=2.0):
    ```python
    ...
    peak_learning_rate = 0.1
    low_learning_rate = 0.01
    total_steps = 1000
    total_steps = 10000
    learning_rate_fn = LinearEpochGradualWarmupPolynomialDecayLearningRate(
        low_learning_rate,
        peak_learning_rate,
        warmup_steps,
        total_steps,
        power=2.0)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_fn),
        loss=‘sparse_categorical_crossentropy’,
        metrics=[‘accuracy’]
        )
    model.fit(data, labels, epochs=5)
    ```
    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer step and outputs the learning rate, a scalar
        `Tensor` of the same type as `low_learning_rate`.
    """
    
    def __init__(
        self,
        low_learning_rate,
        peak_learning_rate,
        warmup_steps,
        total_steps,
        power=2.0,
        name=None):
        """Applies a Linear Epoch Gradual Warmup and Polynomial decay to the learning rate.
        Args:
        low_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
        peak_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The peak learning rate.
        warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the warmup computation above.
        total_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
        power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
        name: String.  Optional name of the operation. Defaults to
            ‘PolynomialDecay’.
        """
        super(LinearEpochGradualWarmupPolynomialDecayLearningRate, self).__init__()
        self.low_learning_rate = low_learning_rate
        # self.learning_rate = tf.convert_to_tensor(low_learning_rate)
        self.learning_rate = tf.Variable(low_learning_rate)
        self.warmup_steps = warmup_steps
        self.peak_learning_rate = peak_learning_rate
        self.total_steps = total_steps
        self.power = power
        self.name = name
        
    def __call__(self, step):
        with tf.name_scope(self.name or "LinearEpochGradualWarmupPolynomialDecayLearningRate") as name:
            low_learning_rate = tf.convert_to_tensor(self.low_learning_rate, name="low_learning_rate")
            dtype = low_learning_rate.dtype
            peak_learning_rate = tf.cast(self.peak_learning_rate, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)
            power = tf.cast(self.power, dtype)
            global_step = tf.cast(step, dtype)
            warmup_percent_done = global_step / warmup_steps
            warmup_learning_rate = (peak_learning_rate - low_learning_rate) * tf.math.pow(
                warmup_percent_done, tf.cast(1.0, dtype)
            ) + low_learning_rate
            total_steps = tf.cast(self.total_steps, dtype)
            decay_steps = total_steps - warmup_steps
            p = tf.divide(
                tf.minimum(global_step - warmup_steps, decay_steps),
                decay_steps
            )
            decay_learning_rate = tf.add(
                tf.multiply(
                    peak_learning_rate - low_learning_rate,
                    tf.pow(1 - p, power)
                ),
                low_learning_rate,
                name="decay_learning_rate"
            )
            learning_rate = tf.cond(
                global_step < warmup_steps,
                lambda: warmup_learning_rate,
                lambda: decay_learning_rate,
                name="learning_rate",
            )
            # self.learning_rate = learning_rate
            self.learning_rate.assign(learning_rate)
            return learning_rate
        
    def get_config(self):
        return {
            "low_learning_rate": self.low_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warm_steps": self.warm_steps,
            "total_steps": self.total_steps,
            "power": self.power,
            "name": self.name
        } 






if __name__ == "__main__":
    with open("./config/model_config/dense_experts_simplex_gate.json") as f:
        config_model = json.load(f)
    # _convert_params_recursive(config_model)
    # print(config_model)
    with open("./config/task_config/task_configs.yml") as f:
        config_tasks = yaml.load(f,Loader=yaml.FullLoader)
    with open("./config/train_config/synthetic_regressions_no_hyperparam_search.json") as f:
        config_train = json.load(f)
    config = MergeConfigs(config_model, config_train, config_tasks)
    # print(config)
    # print("\n")
    # train_dataset, val_dataset = LoadDatasets(config)
    # for batch in train_dataset:
    #     print(batch[0])
    #     print(batch[1])
    #     print(batch[1]["task1"])