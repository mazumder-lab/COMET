"""
File used to run hyperparameter search for the hyperparameters stored in train_config.yml and model_config.json.
Writes results in multiple files, and saves weights for future tests if asked in arguments.
"""
import os
import time
import shutil
import argparse
import json
import pickle
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import optuna
import joblib
from copy import deepcopy
from scipy.special import softmax
import gc

from train_tasks import train_model
from task_utils import (
    MergeConfigs,
    LoadDatasets,
    OptimizerMapper,
    LoadMetrics,
    LoadLosses,
    ForwardModel,
    AssignWeights,
    WeightRecoveryEvaluation,
    LinearEpochGradualWarmupPolynomialDecayLearningRate,
)
from model.main_model import MoE
from model.main_model_stacked import MoEStacked


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    """
    Parses the hyperparameter search command, defines the hyperparameter space, instantiates each model,
    and launches a hyperparameter search (by calling multiple times the train loop defined in train_tasks.py).
    inputs: None
    outputs: None.
    """
    # We define the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        default="./config/model_config/dense_experts_trimmed_lasso_simplex_gate.json",
        type=str,
        help="Path to the model's config (in which its architecture is defined with hyperparameters to tune).",
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
        help="Path to the train config file, in which training hyperparameters are defined.",
    )
    parser.add_argument(
        "--task_config",
        default="./config/task_config/task_configs.yml",
        type=str,
        help="Path to the task configs file, in which task parameters are defined.",
    )
    parser.add_argument(
        "--results_location",
        default="../results/",
        type=str,
        help="Path to location where to save results (weights and performance metrics).",
    )
    parser.add_argument(
        "--experiment_name",
        default="hyperparameter_tuning_"+str(int(time.time())),
        type=str,
        help="Name of the folder where to save results (located in results_location).",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed to fix for reproducibility.",
    )
    parser.add_argument(
        "--weight_saving_format",
        default="tf",
        type=str,
        help="Extension to append when saving weights.",
    )
    parser.add_argument(
        "--no_weight_saving",
        default=False,
        action='store_true',
        help="Whether or not to save the model weights at the end of every trial.",
    )
    parser.add_argument(
        "--max_hyperparameter_evals",
        default=15,
        type=int,
        help="Number of hyperparameter trials to conduct.",
    )
    parser.add_argument(
        "--perform_weight_recovery_analysis",
        default=True,
        type=bool,
        help="Whether to use the ground_truth_weights_location mentioned to perform weight recovery analysis.",
    )
    args = parser.parse_args()
    
    global max_hyperparameter_evals
    max_hyperparameter_evals = args.max_hyperparameter_evals

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # We fix the seeds for reproducibility of the experiments; a seed value can be selected in the arguments of the command.
    random.seed(args.seed)
    tf.random.set_seed(seed=args.seed)

    # We open, process, and add info to the config files (used to define the model architecture, 
    # the training hyperparameters, and the characteristics of the set of tasks to solve).
    global model_config   # global needed to access this in the optuna trial wrapper
    global train_config
    with open(args.model_config) as f:
        print("Loading model config...")
        model_config = json.load(f)
    with open(args.train_config) as f:
        print("Loading train config...")
        train_config = json.load(f)
    with open(args.task_config) as f:
        print("Loading task config...")
        task_config = yaml.load(f,Loader=yaml.FullLoader)
    
    train_config["num_experts"] = model_config["experts"][0]["instances"]
    # We retrieve information (about the taskset considered) in the task_config object
    taskset_id = train_config["taskset_id"]
    relevant_tasks = None
    for taskset in task_config:
        if task_config[taskset]["taskset_id"] == taskset_id:
            relevant_tasks = task_config[taskset]
    if relevant_tasks == None:
        raise ValueError("taskset_id mentioned in train config file not found among all task configs.") 
    tasks = []
    for task in relevant_tasks["tasks"]:
        if task["instances"] > 1:
            tasks += [
                {
                    k: (
                        task[k] if k != "name" else task[k]+str(i+1)
                    ) for k in set(list(task.keys())) - set(["instances"])
                } for i in range(task["instances"])
            ]
        elif task["instances"] == 1:
            tasks.append(
                {
                    k: task[k] for k in set(list(task.keys())) - set(["instances"])
                } 
            )
    task_names = [task["name"] for task in tasks]
    assert(all([task_names[i] != task_names[i+1] for i in range(len(task_names)-1)]))
    if "task_weights" not in train_config.keys():
        train_config["task_weights"] = [1 / len(tasks) for _ in tasks]
    assert(len(tasks) == len(train_config["task_weights"]))
    relevant_tasks["tasks"] = tasks
    train_config["taskset"] = relevant_tasks

    # We add information parsed by the parser to our configs, which contain the hyperparameters to sample in the trials.
    model_config["from_pretrained"] = args.from_pretrained
    train_config["no_weight_saving"] = args.no_weight_saving
    model_config["use_MoE_stacked"] = args.use_MoE_stacked

    # We configure the path/folder where we will store our trial results.
    results_path = args.results_location + (
        "/" if args.results_location[-1] != "/" else ""
    ) + args.experiment_name
    print("\nCreating directory '% s' for performance records and weights..." % results_path)
    abs_results_path = os.path.join(ROOT_PATH, results_path)
    print(abs_results_path)
    if os.path.exists(abs_results_path):
        shutil.rmtree(abs_results_path)
    os.mkdir(abs_results_path)
    os.mkdir(abs_results_path + "/weights")
    os.mkdir(abs_results_path + "/performance")
    train_config["saving_bpath"] = abs_results_path 

    all_results = []
    with open(train_config["saving_bpath"] + "/performance/all_results.json", "w") as f:
        json.dump(
            all_results, 
            f,
            indent=4
        )    
    
    # We start the optuna study where we try to minimize a validation quantity (can be aggregation of task losses, or metrics)
    # and also maximize validation (or train) sparsity.
    global trial_nb
    trial_nb = 0

    train_dataset, val_dataset, test_dataset = LoadDatasets(train_config)
#     test_dataset = LoadDatasets(train_config, trainval=False)
    
    study = optuna.create_study(
        directions=["minimize","maximize"], 
        sampler=optuna.samplers.RandomSampler(seed=args.seed)
    )
    study.optimize(
        lambda trial: trial_wrapper(trial, train_dataset, val_dataset, test_dataset),    # this function defines what we do during a trial (i.e. a single training)
        n_trials=args.max_hyperparameter_evals, 
        show_progress_bar=True,
        callbacks=[logging_callback]
    )
    
    # We retrieve the best results of our hyperparameter search 
    # (those that minimize validation loss (e.g.) and maximize validation sparsity).
    _, best_trial = min(enumerate(study.best_trials), key=lambda x: x[1].values[0])
    best_trial = best_trial.user_attrs
    print(f"\n\nEND OF ALL TRIALS; best hyperparameter search loss: {best_trial['loss']}\n")

    # In addition to the all_results.json list file produced, we also save a summary json file of the best results observed:
    with open(train_config["saving_bpath"] + "/performance/best_trial_results.json", "w") as f:
        json.dump(
            {
                "best_val_loss_hyperparam_search": best_trial['loss'],
                **best_trial['results']
            }, 
            f,
            indent=4
        )



def trial_wrapper(trial, train_dataset, val_dataset, test_dataset):
    """
    Wraps the training loop from the train_tasks.py file in the right format so as to execute it during optuna trials.
    inputs:
        - optuna trial object containing the hyperparameters to sample
    outputs:
        - aggregated val quantity for hyperparam search (= the validation quantity monitored to retrieve the best trials)
        - sparsity measurement (can be val or train)
    """
    # We clear keras session to save memory.
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Needed to be able to keep the same syntax as hyperopt but with optuna.
    _model_config, _train_config = deepcopy(model_config), deepcopy(train_config)
    config = MergeConfigs(_model_config, _train_config, trial)
    
    global trial_nb
    trial_nb = trial_nb + 1

    
    # We load the datasets used for our training (and testing in case we do not save weights along the way).
    print("\nLoading datasets...")
#     train_dataset, val_dataset = LoadDatasets(config)
#     train_dataset_size = sum([next(iter(batch[1].items()))[1].shape[0] for batch in train_dataset])
    train_dataset_size = train_dataset.cardinality().numpy()
    print("\n==========================train_dataset size:", train_dataset_size)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(config['train_batch_size'])
    val_dataset = val_dataset.batch(config['val_batch_size'])

#     if config["no_weight_saving"]:
#     test_dataset = LoadDatasets(config, trainval=False)
    test_dataset = test_dataset.batch(config['val_batch_size'])
    train_steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    print("\n==========================train_steps_per_epoch:", train_steps_per_epoch)

    print("==========nb_epochs:", config["nb_epochs"])
    print("==========learning_rate:", config["optimizer"]["params"]["learning_rate"])

#     for gate in config["gates"]:
#         if gate["module"] == "SoftKTreesEnsembleLearnPermutedGate":
#             gate["params"]["steps_per_epoch"] = int(train_steps_per_epoch)
#             gate["params"]["epochs_for_learning_permutation"] = int(0.75*config["nb_epochs"])

    for perm in config["permutations"]:
        if perm["module"] == "LearnPermutations":
            perm["params"]["steps_per_epoch"] = int(train_steps_per_epoch)
            if perm["params"]["epochs_per_for_learning_permutation"] is not None:
                if perm["params"]["epochs_for_learning_permutation"] is None:
                    perm["params"]["epochs_for_learning_permutation"] = int(perm["params"]["epochs_per_for_learning_permutation"]*config["nb_epochs"])
    
    if config["load_balancing"]["module"] == "KLDivLoadBalancing":
        if config["load_balancing"]["params"]["scheduler"]: 
            config["load_balancing"]["params"]["steps_per_epoch"] = int(train_steps_per_epoch)
            if config["load_balancing"]["params"]["epochs_per_for_load_balance_warm_up"] is not None:
                config["load_balancing"]["params"]["epochs_for_load_balance_warm_up"] = int(config["load_balancing"]["params"]["epochs_per_for_load_balance_warm_up"]*config["nb_epochs"])
            config["load_balancing"]["params"]["total_epochs"] = 20
                

    # We start the trial.
    print(f"\n\n\n\n#################################### Trial {trial_nb} ####################################")
    print(f"\nConfig for trial {trial_nb}:")
    print(
        json.dumps(
            {k: config[k] for k in set(list(config.keys())) - set(["taskset","task_weights"])}, 
            indent=4
        )
    )
    
    # We instantiate our model to train.
    print("\nInstantiating model...")
    if config["use_MoE_stacked"]:
        model = MoEStacked(config)
    else:
        model = MoE(config)
    
    # If our set of tasks to solve requires to load pretrained weights and restore them on our freshly initialized model.
    if config["taskset"]["taskset_id"] in {7,8,12}:
        # We first build our instantiated model.
        for batch in train_dataset:
            model.build(batch[0].shape)
            break
        # Then, we load the pretrained weights.
        relpath = (
            "./data/raw/instance_specific_large_scale_synthetic_regressions_frozen_experts/shared_features/expert_weights.pkl"
            if config["taskset"]["name"] == "instance_specific_large_scale_synthetic_regressions_frozen_experts"
            else (
                "./data/raw/large_scale_synthetic_regressions_frozen_experts/shared_features/expert_weights.pkl"
                if config["taskset"]["name"] == "large_scale_synthetic_regressions_frozen_experts"
                else "./data/raw/synthetic_regressions_frozen_experts/shared_features/expert_weights.pkl"
            )
        )
        experts_weights_abspath = os.path.join(ROOT_PATH, relpath)
        with open(experts_weights_abspath, "rb") as f:
            weights = pickle.load(f)
        # Here, we restore the pretrained weights on our built model.
        if not config["use_MoE_stacked"]:
            for i,expert in enumerate(model.experts):
                expert.set_weights(weights[i])
        else:
            concatenated_weights = np.concatenate([weight[0] for weight in weights], axis=1)
            concatenated_biases = np.concatenate([weight[1] for weight in weights])
            model.experts.set_weights([concatenated_weights, concatenated_biases])
        # We freeze the weights at the end.
        model.experts.trainable = False
        model.bottom.trainable = False
    # But if, instead of selecting a special treatment related to a specific taskset, 
    # we simply want to continue a hyperparameter search from a checkpoint, 
    # we can use the argument "from_pretrained" in the parser.
    elif config["from_pretrained"]:
        for batch in train_dataset:
            model.build(batch[0].shape)
            break
        model.load_weights(config["from_pretrained"])
        try:
            print("ITERATIONS SO FAR: ", model.gates[0].iterations)
        except:
            pass

    # We instantiate the task losses and metrics.
    print("\nInstantiating losses and metrics...")
    task_losses_list = LoadLosses(config)
    task_metrics_list = LoadMetrics(config)

    # In this section, we instantiate the optimizer.
    print("\nInstantiating optimizer...")
    # If a learning rate scheduler is mentioned in the model config, we initialize it here.
    if train_dataset_size % config["train_batch_size"] == 0:
        nb_steps_per_epoch = train_dataset_size / config["train_batch_size"]
    else:
        nb_steps_per_epoch = int(train_dataset_size / config["train_batch_size"]) + 1
    if "learning_rate_scheduling" in config["optimizer"].keys() and config["optimizer"]["learning_rate_scheduling"]:
        max_steps = nb_steps_per_epoch * config["optimizer"]["learning_rate_scheduling"]["nb_epochs_before_min"]
        lr_schedule = LinearEpochGradualWarmupPolynomialDecayLearningRate(
            config["optimizer"]["learning_rate_scheduling"]["reduction_factor"] * config["optimizer"]["params"]["learning_rate"], 
            config["optimizer"]["params"]["learning_rate"],
            config["optimizer"]["learning_rate_scheduling"]["warmup_duration"] * nb_steps_per_epoch, 
            max_steps,
            power=2.0
        )
        config["optimizer"]["params"]["learning_rate"] = lr_schedule
    # Once the learning rate scheduler has been precisely added to the config, we finally instantiate the optimizer.
    optimizer = OptimizerMapper(config["optimizer"]["type"])(
        **config["optimizer"]["params"]
    )

    start_time_train = time.time()
    # We execute our custom training loop here, which returns a lot of monitoring objects.
    config["trial_nb"] = trial_nb
    (
        model, 
        metrics_all_epochs_train, 
        metrics_all_epochs_val, 
        batch_loss_all_epochs_train, 
        batch_loss_all_epochs_val,
        sparsity_all_epochs_train,
        sparsity_all_epochs_val,          
        train_sparsity_at_best_val,
        _,
        best_epoch
    ) = train_model(
        model,
        optimizer, 
        train_dataset, 
        val_dataset, 
        task_losses_list,
        task_metrics_list,
        config,
        config["saving_bpath"],
        save_weights_every_epoch=False   # we won't save the weights at each epoch in hyperparam searches... too expensive
    )    
    end_time_train = time.time()
    hours_train, rem_train = divmod(end_time_train-start_time_train, 3600)
    minutes_train, seconds_train = divmod(rem_train, 60)
    
    # we store our monitoring objects in a dict to facilitate saving in json files along the way.
    metrics = {
        "train": metrics_all_epochs_train,
        "val": metrics_all_epochs_val,
    }
    loss = {
        "train": batch_loss_all_epochs_train,
        "val": batch_loss_all_epochs_val
    }
    sparsity = {
        "train": sparsity_all_epochs_train,
        "val": sparsity_all_epochs_val
    }
    
    val_loss_hyperparam_search = 0

    # We re-run validation at the end of the trial, with the best weights restored by the EarlyStopping callback
    # (we could have retrieved from our logs the best val loss and metric values too...).
    batch_val_losses = []
    # The states of the loss and metric objects have been reinitialized at the last eval in the training loop.
    
    for batch in val_dataset:
        # Forward pass and update val metrics.
        _, batch_loss = ForwardModel(
            model,
            batch,
            task_losses_list, 
            task_metrics_list, 
            training=False
        )
        batch_val_losses.append(float(batch_loss))

    val_loss = sum(
        next(iter(batch[1].items()))[1].shape[0]*batch_val_loss for batch,batch_val_loss in zip(val_dataset, batch_val_losses)
    ) / sum(
        [next(iter(batch[1].items()))[1].shape[0] for batch in val_dataset]  # total size of val_dataset
    )
    # No need to recompute val sparsity at best val loss checkpoint: we retrieved it from our logs.

    # We also re-compute and save the val metric values at best val loss checkpoint.
    best_metrics_val_list = []
    for i,task_metrics in enumerate(task_metrics_list):
        best_task_metrics_val = {}
        for metric in task_metrics:
            metric_value = float(task_metrics[metric].result())
            best_task_metrics_val[metric] = metric_value
            try:
                task_metrics[metric].reset_state()
            except:
                task_metrics[metric].reset_states()
        best_metrics_val_list.append(best_task_metrics_val)
    best_task_metrics_val = {}
    for met in model.metrics:
        best_task_metrics_val[met.name] = float(met.result())
        met.reset_states()
    best_metrics_val_list.append(best_task_metrics_val)

    # Here, once we have re-computed the val loss and val metrics at best val loss checkpoint, 
    # we now compute the val hyperparam search quantity monitored by optuna.
    for i,task in enumerate(config["taskset"]["tasks"]):
        val_loss_hyperparam_search += config["task_weights"][i] * (
            val_loss if task["hyperparam_search_metric"] == "Loss" 
            else float(
                best_metrics_val_list[i][task["hyperparam_search_metric"]]
            )
        )
    # No need to reset the states of the metrics bc we will re-instantiate them at the next trial

    # Optuna and hyperopt do not deal well with NaNs returned by TF => convert then to Inf.
    if np.isnan(val_loss_hyperparam_search):
        val_loss_hyperparam_search = np.inf        

    # A simple check to see whether gate weights all sum up to one (only for DSelect-K for now).
    # if "enforce_simplex_tol" in config.keys() and config["enforce_simplex_tol"] != None:
    #     tol = config["enforce_simplex_tol"]
    #     try:
    #         for gate in model.gates:
    #             if "d_select_k" in gate.name:
    #                 weights = gate.gate._compute_expert_weights()[0].numpy()
    #                 if abs(sum(weights) - 1.0) > tol:
    #                     # invalid result obtained in the trial
    #                     val_loss_hyperparam_search = np.inf
    #                     break
    #     except:
    #         pass

    print(f"End of trial; best hyperparameter search loss: {val_loss_hyperparam_search}")
    
                    
    # We run testing at the end of the trial in case we do not want to save the weights of the model at the end of each trial.
#     if config["no_weight_saving"]:
    batch_test_losses = []
    batch_test_sparsities = []

    for batch in test_dataset:
        # Forward pass and update val metrics
        _, batch_loss = ForwardModel(
            model,
            batch,
            task_losses_list, 
            task_metrics_list, 
            training=False
        )
        batch_test_losses.append(float(batch_loss))

        # We save the test sparsity along the way.
        try:     
            # when gates not stacked
            batch_test_sparsity = [] 
            for i,gate in enumerate(model.gates):
                if len(gate.metrics) > 0: 
                    for gm in gate.metrics:
                        if gm.name == 'avg_sparsity':
                            batch_test_sparsity.append(gm.result())
                            break
            batch_test_sparsities.append(sum(batch_test_sparsity) / len(batch_test_sparsity))
        except:
            # when gates stacked
            if len(model.gates.metrics) > 0: 
                for gm in model.gates.metrics:
                    if gm.name == 'avg_sparsity':
                        batch_test_sparsities.append(gm.result()) 
                        break 

    test_loss = sum(
        next(iter(batch[1].items()))[1].shape[0]*batch_test_loss for batch,batch_test_loss in zip(test_dataset, batch_test_losses)
    ) / sum(
        [next(iter(batch[1].items()))[1].shape[0] for batch in test_dataset]
    )
    print(f"Test loss at best val: {test_loss}")

    test_sparsity = sum(
        next(iter(batch[1].items()))[1].shape[0]*batch_test_sparsity for batch,batch_test_sparsity in zip(test_dataset, batch_test_sparsities)
    ) / sum(
        [next(iter(batch[1].items()))[1].shape[0] for batch in test_dataset] 
    )
    print(f"Test sparsity at best val: {test_sparsity}")
    sparsity["test"] = float(test_sparsity)

    # We compute and save the test metric values at best val loss checkpoint.
    best_metrics_test_list = []
    for i,task_metrics in enumerate(task_metrics_list):
        if (
            'display_task_metrics_each_epoch' not in config.keys() 
            or config['display_task_metrics_each_epoch'] == True
        ):
            print(f"\t Test metrics for Task {config['taskset']['tasks'][i]['name']}")
        best_task_metrics_test = {}
        for metric in task_metrics:
            metric_value = float(task_metrics[metric].result())
            best_task_metrics_test[metric] = metric_value
            if (
                'display_task_metrics_each_epoch' not in config.keys() 
                or config['display_task_metrics_each_epoch'] == True
            ):
                print(f"\t\t Test {metric} over trial: {metric_value}")
            try:
                task_metrics[metric].reset_state()
            except:
                task_metrics[metric].reset_states()
        best_metrics_test_list.append(best_task_metrics_test)
    best_task_metrics_test = {}
    for met in model.metrics:
        print(met.name, met.result())
        best_task_metrics_test[met.name] = float(met.result())
        met.reset_states()
    best_metrics_test_list.append(best_task_metrics_test)
        
    if (
        'display_task_metrics_each_epoch' not in config.keys() 
        or config['display_task_metrics_each_epoch'] == True
    ):
        print("\n")            

    # We create a doc object that contains all our measurements for this trial.
    # Here, in the case where we do not save the model weights at the end of each trial, we also save 
    # the measurements made on the test set.
    
    trial_results = {
        'trial_nb': trial_nb,
        'best_hyperparam_search_loss_val': val_loss_hyperparam_search,
        'best_metrics_val': best_metrics_val_list,
        'best_test_loss': test_loss,
        'best_metrics_test': best_metrics_test_list,
        'config': config,
        'num_of_params': model.count_params(),
        'start_time_train': start_time_train,
        'end_time_train': end_time_train,
        'training_duration': "{:0>2}:{:0>2}:{:05.2f}".format(int(hours_train),int(minutes_train),seconds_train),
        'metrics': metrics,
        'loss': loss,
        'sparsity': sparsity,
        'best_epoch': best_epoch
    }

#     else:
#         print("\n")
#         # We create a doc object that contains all our measurements for this trial.
#         # No test set measurements here.
#         trial_results = {
#             'trial_nb': trial_nb,
#             'best_hyperparam_search_loss_val': val_loss_hyperparam_search,
#             'best_metrics_val': best_metrics_val_list,
#             'config': config,
#             'metrics': metrics,
#             'loss': loss,
#             'sparsity': sparsity,
#             'best_epoch': best_epoch
#         }


    # One last case before the end of the trial: 
    # if our taskset involves a weight recovery evaluation (i.e.: how do the weights of our trained model compare to that of 
    # a ground truth data generating function), then we perform the weight recovery analysis here.
    if config["taskset"]["taskset_id"] == 12:
        if config["use_MoE_stacked"]:
            print("##### WARNING: use of stacked experts and gates for weight recovery eval not supported yet #####")
        else:
            if config["gates"][0]["params"]["use_routing_input"]:
                print("##### WARNING: weight recovery eval on instance-specific gates not supported yet #####")
            else:
                # Using this on any gate other than (sparse) simplex, trimmed lasso simplex, softmax, topk softmax, dselect-k
                # will trigger an error.
                print("Weight recovery evaluation:")
                # We first load the ground truth weights we expect to have recovered.
                with open(
                    "./data/raw/synthetic_regressions_frozen_experts/shared_features/gate_weights.pkl",
                    "rb"
                ) as f:
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

                    elif "softmax" in gate.name and ("topk" not in gate.name and "trimmed_lasso" not in gate.name):
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
                
                # We add our weight recovery measurements to the trial_results dict.
                trial_results["weight_recovery"] = weight_recovery_eval
    
    
    # Before finishing the trial, we save our trial_results dict in the all_results.json list file.
    try:
        with open(config["saving_bpath"] + "/performance/all_results.json", "r") as f:
            all_results = json.load(f)    
        with open(config["saving_bpath"] + "/performance/all_results.json", "w") as f:
            json.dump(
                all_results + [trial_results], 
                f,
                indent=4
            ) 
    except:
        pass

    trial.set_user_attr('loss', val_loss_hyperparam_search)
    trial.set_user_attr('results', trial_results)
#     if not config["no_weight_saving"]:
#         trial.set_user_attr('model', model)

    return val_loss_hyperparam_search, sparsity_all_epochs_val[best_epoch-1]



def logging_callback(study, frozen_trial, save_freq=5):
    """
    Callback function executed at the end of each trial to save information on the disk.
    inputs:
        - optuna study which we enrich with our trials
        - the previous trial
        - the frequency at which we save our information on the disk
    outputs:
        - None
    """
    # We print in our logs the best val quantity observed so far, just to keep track.
    if "previous_best_value" not in study.user_attrs.keys():
        study.set_user_attr("previous_best_value", frozen_trial.values[0])
        print(" => Best:", frozen_trial.values[0])
    else:
        previous_best_value = study.user_attrs["previous_best_value"]
        if previous_best_value > frozen_trial.values[0]:
            study.set_user_attr("previous_best_value", frozen_trial.values[0])
            print(" => Best:", frozen_trial.values[0])
        else:
            print(" => Best:", previous_best_value)
    
    # We save a csv of the study every 5 trials 
    # AND also a Pareto front graph of our val loss-val sparsity hyperparam search 
    # (plus some hyperparameter importance graphs for each of the 2 objectives).
    if trial_nb % save_freq or trial_nb >= max_hyperparameter_evals-1:
        # saving of the csv of the study
        try: 
            joblib.dump(study, train_config["saving_bpath"] + "/performance/study.pkl")  
            df = study.trials_dataframe()
            df.to_csv(train_config["saving_bpath"] + "/performance/hyperparam_results.csv")
        except:
            pass
        # saving of the Pareto frontier and the hyperparameter importance graphs
        try:
            fig1 = optuna.visualization.plot_pareto_front(study, target_names=["hyperparam_search_loss","sparsity"])
            fig1.write_image(file=train_config["saving_bpath"] + "/performance/pareto_front.png", format="png")
            fig2 = optuna.visualization.plot_param_importances(
                study, target=lambda t: t.values[0], target_name="hyperparam_search_loss"
            )
            fig2.write_image(file=train_config["saving_bpath"] + "/performance/hyperparam_importance_loss.png", format="png")
            fig3 = optuna.visualization.plot_param_importances(
                study, target=lambda t: t.values[1], target_name="sparsity"
            )
            fig3.write_image(file=train_config["saving_bpath"] + "/performance/hyperparam_importance_sparsity.png", format="png")     
        except:
            pass



if __name__ == "__main__":
    # from importlib import reload
    # reload(tf.keras.models)
    main()
    # model = tf.keras.models.load_model("/Users/mathieusibue/MBAn/RA/code/results/test_softmax_hyperparam_search/weights/tf_whole_model_1")
    # model.summary()
