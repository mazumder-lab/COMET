"""
File implementing a single forward pass on the test set.
+ provides a main function with argument parsing to run the file as a script if needed!
Saves results in a file.
WARNING: some features might be deprecated because of the script has not been used in a long time.
"""
import numpy as np
import tensorflow as tf
import json
import os
import argparse
import shutil
import yaml
import time
import pickle
from scipy.special import softmax
from task_utils import (
    MergeConfigs,
    ForwardModel,
    LoadDatasets,
    LoadMetrics,
    LoadLosses,
    AssignWeights,
    WeightRecoveryEvaluation
)
from model.gates.dselect_k_gate import DSelectKGate
from model.main_model import MoE
from model.main_model_stacked import MoEStacked


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    """
    Parses the arguments from the command and launches a test procedure using test_model.
    inputs: None
    outputs: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        default="./config/model_config/dense_experts_trimmed_lasso_simplex_gate.json",
        type=str,
        help="Path to the model's config (in which its architecture is defined).",
    )
    parser.add_argument(
        "--from_pretrained",
        default=None,
        type=str,
        help="Path to pretrained weights of the model architecture defined in the config.",
    )
    parser.add_argument(
        "--train_config",
        default="./config/train_config/example_train.json",
        type=str,
        help="Path to train config (also containing val parameters to be used for testing).",
    )
    parser.add_argument(
        "--task_config",
        default="./config/task_config/task_configs.yml",
        type=str,
        help="Path to task configs in which task parameters are defined.",
    )
    parser.add_argument(
        "--results_location",
        default="../results/",
        type=str,
        help="Path to location where to save results (performance metrics).",
    )
    parser.add_argument(
        "--experiment_name",
        default="test_"+str(int(time.time())),
        type=str,
        help="Name of the folder where to save results (located in results_location).",
    )
    parser.add_argument(
        "--ground_truth_weights_location",
        default="./data/raw/synthetic_regressions/shared_features/gate_weights.pkl",
        type=str,
        help="Path to optimal weights for support recovery evaluation.",        
    )
    parser.add_argument(
        "--use_test_dataset",
        default=True,
        type=bool,
        help="Whether to use the test or val dataset for out-of-sample evaluation.",
    )
    parser.add_argument(
        "--perform_weight_recovery_analysis",
        default=False,
        action='store_true',
        help="Whether to use the ground_truth_weights_location mentioned to perform weight recovery analysis.",
    )
    parser.add_argument(
        "--use_MoE_stacked",
        default=False,
        action='store_true',
        help="Whether or not to use the MoE implementation with stacked experts and gates (so long as they are all of the same type).",
    )
    args = parser.parse_args()

    # We open, merge, and add info to the config files (used to define the model architecture, 
    # the training hyperparameters, and the characteristics of the set of tasks to solve).
    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.train_config) as f:
        train_config = json.load(f)
    with open(args.task_config) as f:
        task_config = yaml.load(f,Loader=yaml.FullLoader)
    config = MergeConfigs(model_config, train_config, task_config)
    config['ground_truth_weights_location'] = args.ground_truth_weights_location
    config['perform_weight_recovery_analysis'] = args.perform_weight_recovery_analysis
    config['use_MoE_stacked'] = args.use_MoE_stacked

    # We load the datasets used for our tests.
    # NOTE: either the test dataset OR val dataset from trainval can be used.
    print("Loading datasets...")
    if args.use_test_dataset:
        test_dataset = LoadDatasets(config, trainval=False)
    else:
        _, test_dataset = LoadDatasets(config)
    
    # We instantiate our model to train.
    print("\nInstantiating model...")
    if config['use_MoE_stacked']:
        model = MoEStacked(config)
    else:
        model = MoE(config)

    # We instantiate the task losses and metrics.
    print("Instantiating losses and metrics...")
    task_losses_list = LoadLosses(config)
    task_metrics_list = LoadMetrics(config)

    # We configure the path/folder where we will store our test results.
    results_path = args.results_location + args.experiment_name
    print("Creating directory '% s' for performance records..." % results_path)
    abs_results_path = os.path.join(ROOT_PATH, results_path)
    if os.path.exists(abs_results_path):
        # os.mkdir(abs_results_path)
        shutil.rmtree(abs_results_path)
    os.mkdir(abs_results_path)
    # shutil.rmtree(abs_results_path)
    os.mkdir(abs_results_path + "/performance")

    # We execute once our custom testing loop here (defined below), which returns monitoring objects.
    (
        metrics_test, 
        batch_loss_test,
        weight_recovery_eval
    ) = test_model(
        model, 
        test_dataset, 
        task_losses_list,
        task_metrics_list,
        config
    )

    # we store our monitoring objects in a dict to facilitate saving in json files.
    metrics = {
        "test": metrics_test,
        "weight_recovery_eval": weight_recovery_eval
    }
    loss = {
        "test": batch_loss_test,
    }

    # We save our monitoring objects.
    with open(abs_results_path + "/performance/metrics.json", "w") as f:
        json.dump(metrics,f)
    with open(abs_results_path + "/performance/loss.json", "w") as f:
        json.dump(loss,f)
    return


def test_model(
    model, 
    test_dataset, 
    task_losses_list,
    task_metrics_list,
    config
):
    """
    Computes the metrics of the model for each task, when tested on the test set (or the val set).
    Performs weight recovery on unstacked static gates if requested in config.
    inputs:
        - instantiated model to train (does not need to be built yet, unless pretrained)
        - test (or val) dataloader
        - list of task losses objects to aggregate at the end of each forward pass
        - list of task metrics objects whose value we need to save at the end of each forward pass
        - merged config with training-related hyperparameter values 
    outputs:
        - metrics computed for all epochs on the test (or val) set 
        - mini-batch loss computed for each mini-batch on the test (or val) set 
        - weight recovery evaluation results (if requested in config).
    """
    # We instantiate the objects we will use to store the measurements we will make during the tests.
    metrics_test = [
        {
            metric: 0
            for metric in metrics_dict
        }
        for metrics_dict in task_metrics_list
    ]  
    batch_loss_test = [] 

    # We save our test set size for convenience
    test_dataset_size = sum([batch[0].shape[0] for batch in test_dataset])

    start_time = time.time()

    # In our implementation, all gates have to be either instance-specific or static; not a mix of both.
    print("\nStart of test procedure")
    # Iterate over the batches of the test dataset.
    for batch in test_dataset:
        # Forward pass and update test metrics
        _, batch_loss = ForwardModel(
            model,
            batch,
            task_losses_list,
            task_metrics_list,
            training=False
        )
        
        # We store batch-wise measurements here (in our objects).
        batch_loss_test.append(float(batch_loss))

    print("Time taken: %.2fs\n\n" % (time.time() - start_time))

    # Aggregated metrics on the test set for this single epoch, based on our measurements for each step.
    mean_loss = sum(
        batch[0].shape[0] * batch_loss for batch,batch_loss in zip(test_dataset,batch_loss_test)
    ) / test_dataset_size
    print(f"Test loss: {mean_loss}\n")

    # We display test metrics at the end of this epoch.
    for i,task_metrics in enumerate(task_metrics_list):
        print(f"--- Test metrics for Task {config['taskset']['tasks'][i]['name']} ---")
        for metric in task_metrics:
            metric_value = float(task_metrics[metric].result())
            metrics_test[i][metric] = metric_value
            print(f"-------- Test {metric}: {metric_value} --------\n")

            
    # If our taskset involves a weight recovery evaluation (i.e.: how do the weights of our trained model compare to that of 
    # a ground truth data generating function), then we perform the weight recovery analysis here.
    if (
        config["perform_weight_recovery_analysis"]
    ) and (
        (
            not config['use_MoE_stacked'] and not model.gates[0].use_routing_input
        )
    ):
        # Using this on any gate other than (sparse) simplex, trimmed lasso simplex, softmax, topk softmax, dselect-k
        # will trigger an error.
        print("Weight recovery evaluation:")
        # We first load the ground truth weights we expect to have recovered.
        with open(config['ground_truth_weights_location'],"rb") as f:
            all_ground_truth_weights = [
                ground_truth_weights.reshape(1,-1) for ground_truth_weights in pickle.load(f)
            ]
        all_ground_truth_weights = np.concatenate(all_ground_truth_weights)

        # Then we extract the learned weights of the gates of our model.
        # Some of our types of static gates need their underlying weights to be reparameterized in order to be compared
        # to the ground truth.
        all_gate_weights = []
        for i,gate in enumerate(model.gates):
            gate_weights = gate.get_weights()[0]

            if "softmax" in gate.name and "top_k" in gate.name:
                topk = tf.math.top_k(
                    tf.reshape(tf.expand_dims(gate_weights, 1), [-1]),
                    k=gate.get_config()["k"]
                )
                topk_scattered = tf.scatter_nd(
                    tf.reshape(topk.indices, [-1, 1]),
                    topk.values,
                    [gate_weights.shape[0]]
                )
                topk_prep = tf.where(
                    tf.math.equal(topk_scattered, tf.constant(0.0)),
                    -np.inf*tf.ones_like(topk_scattered),  # we add the mask here
                    topk_scattered
                )
                gate_weights = tf.nn.softmax(
                    tf.expand_dims(topk_prep, 1),  # else, we get an error in the softmax activation
                    axis=0
                ).numpy()

            elif "softmax" in gate.name and "topk" not in gate.name:
                gate_weights = softmax(
                    gate_weights
                )
            
            elif "d_select_k" in gate.name:
                gate_weights = gate.gate.compute_expert_weights()[0].numpy()

            gate_weights = gate_weights.reshape(1,-1)

            print(f"--- Weights of gate for {config['taskset']['tasks'][i]['name']}: ---")
            print(gate_weights)
            print("\n")
            all_gate_weights.append(gate_weights)

        all_gate_weights = np.concatenate(all_gate_weights)

        # Now that we have loaded the ground truth weights 
        # AND extracted/re-parameterized the learned weights of our gates, 
        # we finally perform the weight recovery analysis.
        _, perm_predicted_weights = AssignWeights(
            all_gate_weights,
            all_ground_truth_weights
        )
        weight_recovery_eval = WeightRecoveryEvaluation(
            all_ground_truth_weights,
            all_gate_weights[:,perm_predicted_weights],
        )

        return (
            metrics_test, 
            batch_loss_test, 
            weight_recovery_eval
        )

    else:
        return (
            metrics_test, 
            batch_loss_test, 
            "no weight recovery evaluation"
        )

    

if __name__ == "__main__":
    main()