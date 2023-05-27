"""
File implementing a single training loop (with eval at the end of each epoch)
+ provides a main function with parsed arguments to run the file as a script if needed.
Saves weights for each epoch in a different file
+ saves training results and tensorboards in other files.
"""
import os
import sys
import time
import shutil
import argparse
import json
import yaml
import datetime
import tensorflow as tf
from tqdm import tqdm
import random
from task_utils import (
    MergeConfigs,
    ForwardModel,
    LoadDatasets,
    OptimizerMapper,
    CallbackMapper,
    LoadMetrics,
    LoadLosses,
    LinearEpochGradualWarmupPolynomialDecayLearningRate,
)
from model.main_model import MoE
from model.main_model_stacked import MoEStacked


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    """
    Parses the arguments from the command and launches a training loop using train_model.
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
        default="training_"+str(int(time.time())),
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
        "--save_weights_every_epoch",
        default=False,
        type='store_true',
        help="Whether or not to save the model weights at the end of every epoch.",
    )
    parser.add_argument(
        "--weight_saving_format",
        default="tf",
        type=str,
        help="Extension to append when saving weights.",
    )
    parser.add_argument(
        "--use_MoE_stacked",
        default=False,
        action='store_true',
        help="Whether or not to use the MoE implementation with stacked experts and gates (so long as they are all of the same type).",
    )
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # We fix the seeds for reproducibility of the experiments; a seed value can be selected in the arguments of the command.
    random.seed(args.seed)
    tf.random.set_seed(seed=args.seed)

    # We open, merge, and add info to the config files (used to define the model architecture, 
    # the training hyperparameters, and the characteristics of the set of tasks to solve).
    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.train_config) as f:
        train_config = json.load(f)
    with open(args.task_config) as f:
        task_config = yaml.load(f,Loader=yaml.FullLoader)
    config = MergeConfigs(model_config, train_config, task_config)
    config["use_MoE_stacked"] = args.use_MoE_stacked
    config["from_pretrained"] = args.from_pretrained
    
    # We load the datasets used for our training.
    print("\nLoading datasets...")
    train_dataset, val_dataset = LoadDatasets(config)
#     print("\n==========================train_dataset size:", tf.data.experimental.cardinality(train_dataset).numpy())
    
    # We instantiate our model to train.
    print("\nInstantiating model...")
    if config["use_MoE_stacked"]:
        model = MoEStacked(config)
    else:
        model = MoE(config)
        
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
    train_dataset_size = sum([next(iter(batch[1].items()))[1].shape[0] for batch in train_dataset])
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
            config["optimizer"]["learning_rate_scheduling"]["warmup_duration"] * nb_steps_per_epoch,    # warmup duration; initially 5
            max_steps,
            power=2.0
        )
        config["optimizer"]["params"]["learning_rate"] = lr_schedule
    # Once the learning rate scheduler has been precisely added to the config, we finally instantiate the optimizer.
    optimizer = OptimizerMapper(config["optimizer"]["type"])(
        **config["optimizer"]["params"]
    )

    # We configure the path/folder where we will store our training results.
    results_path = args.results_location + (
        "/" if args.results_location[-1] != "/" else ""
    ) + args.experiment_name
    print("\nCreating directory '% s' for performance records and weights..." % results_path)
    abs_results_path = os.path.join(ROOT_PATH, results_path)
    if os.path.exists(abs_results_path):
        shutil.rmtree(abs_results_path)
    os.mkdir(abs_results_path)
    os.mkdir(abs_results_path + "/weights")
    os.mkdir(abs_results_path + "/monitoring")
    os.mkdir(abs_results_path + "/performance")

    # We execute once our custom training loop here (defined below), which returns a lot of monitoring objects.
    (
        model, 
        metrics_all_epochs_train, 
        metrics_all_epochs_val, 
        batch_loss_all_epochs_train, 
        batch_loss_all_epochs_val,
        sparsity_all_epochs_train,
        sparsity_all_epochs_val,        
        _,
        _,
        _
    ) = train_model(
        model,
        optimizer, 
        train_dataset, 
        val_dataset, 
        task_losses_list,
        task_metrics_list,
        config,
        abs_results_path,
        args.save_weights_every_epoch
    )

    # we store our monitoring objects in a dict to facilitate saving in json files.
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
        "val": sparsity_all_epochs_val,
    }

    # We save weights of the model in case we did not save them at each epoch.
    if not args.save_weights_every_epoch:
        print("==================", abs_results_path + "/weights/" + "tf_model_" + str(config["nb_epochs"]) + "." + args.weight_saving_format)
        model.save_weights(
            abs_results_path + "/weights/" + "tf_model_" + str(config["nb_epochs"]) + "." + args.weight_saving_format
        )
#         model.save(
#             abs_results_path + "/weights/" + "tf_whole_model_" + str(config["nb_epochs"]) + "." + args.weight_saving_format
#         )
       
    # We save our monitoring objects.
    with open(abs_results_path + "/performance/metrics.json", "w") as f:
        json.dump(
            metrics, 
            f,
            indent=4
        )
    with open(abs_results_path + "/performance/loss.json", "w") as f:
        json.dump(
            loss, 
            f,
            indent=4
        )
    with open(abs_results_path + "/performance/sparsity.json", "w") as f:
        json.dump(
            sparsity, 
            f,
            indent=4
        )



def train_model(
    model, 
    optimizer, 
    train_dataset, 
    val_dataset, 
    task_losses_list,
    task_metrics_list,
    config,
    weight_saving_path,
    weight_saving_format="tf",
    save_weights_every_epoch=False,
    log_every_k=128
):
    """
    Just trains an instantiated model by looping over the training set for a given nb of epochs, 
    and evaluating it using val set.
    (Note: lots of outputed objects ended up being redundant here)
    inputs:
        - instantiated model to train (does not need to be built yet, unless pretrained)
        - instantiated optimizer to use for the training
        - train dataloader
        - val dataloader
        - list of task losses objects to aggregate at the end of each forward pass
        - list of task metrics objects whose value we need to save at the end of each forward pass
        - merged config with training-related hyperparameter values 
        - path where to save weights (str)
        - format for weight saving (deprecated)
        - option to save weights on disk of the model at the end of each epoch
        - log measurements every k steps
    outputs:
        - train model object
        - metrics computed for all epochs on the train set
        - metrics computed for all epochs on the val set 
        - mini-batch loss computed for all epochs on the train set
        - mini-batch loss computed for all epochs on the val set 
        - gate sparsity computed for all epochs on the train set
        - gate sparsity computed for all epochs on the val set
        - gate sparsity on the train set at best val checkpoint
        - gate sparsity on the val set at best val checkpoint
        - best epoch number in terms of val loss
    """   
    # We instantiate the objects we will use to store the measurements we will make during the training.
    metrics_all_epochs_train = [
        {
            metric: []
            for metric in metrics_dict
        }
        for metrics_dict in task_metrics_list
    ]
    metrics_all_epochs_val = [
        {
            metric: []
            for metric in metrics_dict
        }
        for metrics_dict in task_metrics_list
    ] 

    batch_loss_all_epochs_train = [
        [] for _ in range(config["nb_epochs"])
    ] 
    batch_loss_all_epochs_val = [
        [] for _ in range(config["nb_epochs"])
    ] 
    loss_all_epochs_val = [] 
    
    sparsity_all_epochs_train = [
        [] for _ in range(config["nb_epochs"])
    ] 
    sparsity_all_epochs_val = [
        [] for _ in range(config["nb_epochs"])
    ] 
    
    # We also initialize our Tensorboard logs to save and visualize information in real time.
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = weight_saving_path + "/monitoring/" + (
        current_time if "trial_nb" not in config.keys() else "trial_" + str(config["trial_nb"])
    ) + '/train'
    val_log_dir = weight_saving_path + "/monitoring/" + (
        current_time if "trial_nb" not in config.keys() else "trial_" + str(config["trial_nb"])
    ) + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # We initialize our TF callbacks based on our train_config 
    # (for EarlyStopping and stopping when the train loss diverges/explodes for instance).
    _callbacks = (
        [
            CallbackMapper(callback["type"])(
                **callback["params"]
            ) for callback in config["callbacks"]
        ] if "callbacks" in config.keys() 
        else []
    )
    callbacks = tf.keras.callbacks.CallbackList(
        _callbacks, 
        add_history=True, 
        model=model
    )
    for i,callback in enumerate(callbacks.callbacks):
        if isinstance(callback, tf.keras.callbacks.EarlyStopping) and callback.monitor != "val_loss":
            raise ValueError("EarlyStopping monitoring on a metric other than the val loss not supported yet.")
    logs = {}
    callbacks.on_train_begin(logs=logs)
    
    # We save our train and val set sizes for convenience
    train_batch_size = config["train_batch_size"]
    train_dataset_size = sum([next(iter(batch[1].items()))[1].shape[0] for batch in train_dataset])
    print("=======train_dataset_size:", train_dataset_size)
    val_dataset_size = sum([next(iter(batch[1].items()))[1].shape[0] for batch in val_dataset])
    print("=======val_dataset_size:", val_dataset_size)
    total_steps = 0
    
    
    # Precompiled graph of operations for training
    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            # Update training metrics and compute total loss.
            preds, batch_loss = ForwardModel(
                model,
                batch,
                task_losses_list,
                task_metrics_list,
                training=True
            )
        step_grads = tape.gradient(batch_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(step_grads, model.trainable_weights))

        return preds, batch_loss

    # Precompiled graph of operations for validation and test
    @tf.function
    def val_step(batch):
        # Forward pass and update val metrics
        preds, batch_loss = ForwardModel(
            model,
            batch,
            task_losses_list,
            task_metrics_list,
            training=False
        )
        return preds, batch_loss

    
    show_model_summary = True
    # Iterate over epochs.
    for epoch in range(config["nb_epochs"]):
        print("\n\t\t ---------------- START OF EPOCH %d ----------------\n" % (epoch+1,))
        
        start_time = time.time()

        callbacks.on_epoch_begin(epoch, logs=logs)
        gradients = None    # represents our total gradients, to be able to support gradient accumulation      

        # Iterate over the batches of the train dataset.
        for step, batch in enumerate(train_dataset):
            total_steps += 1
            model.reset_states()
            
            
            if config["gates"][0]["module"] == "TrimmedLassoSimplexProjGate":   
                if "learning_rate_scheduling" in config["optimizer"].keys():
                    learning_rate = optimizer.learning_rate.learning_rate
                    try:
                        for gate in model.gates:
                            gate.learning_rate = learning_rate
                    except:
                        model.gates.learning_rate = learning_rate

            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_train_batch_begin(step, logs=logs)

            # Forward pass and backward pass + optimizer step here.
            _, batch_loss = train_step(batch)
            
            if show_model_summary:
                tf.print(model.summary())
                show_model_summary = False
                
            # We store batch-wise measurements here (in our objects and for the Tensorboard)
            batch_loss_all_epochs_train[epoch].append(float(batch_loss))
            with train_summary_writer.as_default():
                # We save the train batch loss and the learning rate during this step.
                tf.summary.scalar('loss', batch_loss, step=total_steps)
                tf.summary.scalar(
                    'learning_rate', 
                    (
                       optimizer.learning_rate.learning_rate if hasattr(
                           optimizer.learning_rate, 
                           "learning_rate"
                       ) else optimizer.learning_rate
                    ), 
                    step=total_steps
                )
                # We also measure the train batch sparsity during this step and save it.
                try:
                    # when the gates are not stacked
                    batch_train_sparsity = [] 
                    for i,gate in enumerate(model.gates):
                        if len(gate.metrics) > 0: 
                            for gm in gate.metrics:
                                tf.summary.scalar(
                                    f"{config['taskset']['tasks'][i]['name']} {gm.name}",
                                    float(gm.result()),
                                    step=total_steps
                                )
                                if gm.name == 'avg_sparsity':
                                    batch_train_sparsity.append(gm.result())
                    sparsity_all_epochs_train[epoch].append(sum(batch_train_sparsity) / len(batch_train_sparsity))
                except:
                    # when the gates are stacked
                    if len(model.gates.metrics) > 0: 
                        for gm in model.gates.metrics:
                            tf.summary.scalar(
                                f"{config['taskset']['name']} {gm.name}",
                                float(gm.result()),
                                step=total_steps
                            )
                            if gm.name == 'avg_sparsity':
                                sparsity_all_epochs_train[epoch].append(gm.result()) 
                train_summary_writer.flush()

            callbacks.on_train_batch_end(step, logs={"loss": float(batch_loss)})
            callbacks.on_batch_end(step, logs={"loss": float(batch_loss)})

            # Log measurements every log_every_k steps (i.e. batches here).
            if step % log_every_k == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(batch_loss))
                )
                print("Seen so far: %d samples" % ((step + 1) * train_batch_size))

        print(f"\n\t\t----------------- END OF EPOCH {epoch+1} -----------------")

        # Aggregated metrics on the train set for this epoch based on our measurements for each step.
        train_mean_loss = sum(
            next(iter(batch[1].items()))[1].shape[0]*batch_loss for batch,batch_loss in zip(train_dataset,batch_loss_all_epochs_train[epoch])
        ) / train_dataset_size
        print(f"\n\t Train loss over epoch: {train_mean_loss}")
        train_mean_sparsity = sum(
            next(iter(batch[1].items()))[1].shape[0]*batch_sparsity for batch,batch_sparsity in zip(train_dataset,sparsity_all_epochs_train[epoch])
        ) / train_dataset_size
        print(f"\t Avg train sparsity over epoch: {train_mean_sparsity}")
        sparsity_all_epochs_train[epoch] = float(train_mean_sparsity)
        
        # We display and save train metrics at the end of each epoch.
        for i,task_metrics in enumerate(task_metrics_list):
            if (
                'display_task_metrics_each_epoch' not in config.keys() 
                or config['display_task_metrics_each_epoch'] == True
            ):
                print(f"\t Training metrics for Task {config['taskset']['tasks'][i]['name']}")
            for metric in task_metrics:
                metric_value = float(task_metrics[metric].result())
                metrics_all_epochs_train[i][metric].append(metric_value)
                if (
                    'display_task_metrics_each_epoch' not in config.keys() 
                    or config['display_task_metrics_each_epoch'] == True
                ):
                    print(f"\t\t Train {metric} over epoch: {metric_value}")
                # Reset training metrics at the end of each epoch
                try:
                    task_metrics[metric].reset_state()
                except:
                    task_metrics[metric].reset_states()
                with train_summary_writer.as_default():
                    tf.summary.scalar(f"{config['taskset']['tasks'][i]['name']} {metric}", metric_value, step=epoch+1)
                    train_summary_writer.flush()
            if (
                'display_task_metrics_each_epoch' not in config.keys() 
                or config['display_task_metrics_each_epoch'] == True
            ):
                print("\n")
        for met in model.metrics:
            print(met.name)
            met.reset_states()

                
                
        # We run a validation loop at the end of each epoch.
        print(f"\n\t\t --------- RUNNING VALIDATION FOR EPOCH {epoch+1} ---------\n")
        # Iterate over the batches of the val dataset.
        for batch in tqdm(val_dataset):
            model.reset_states()

            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_test_batch_begin(step, logs=logs)

            # Forward pass here.
            _, batch_loss = val_step(batch)

            # We store batch-wise measurements here (in our objects and for the Tensorboard).
            batch_loss_all_epochs_val[epoch].append(float(batch_loss))
            # We measure the val batch sparsity during this step and save it.
            try:     
                # when gates not stacked
                batch_val_sparsity = [] 
                for i,gate in enumerate(model.gates):
                    if len(gate.metrics) > 0: 
                        for gm in gate.metrics:
                            if gm.name == 'avg_sparsity':
                                batch_val_sparsity.append(gm.result())
                                break
                sparsity_all_epochs_val[epoch].append(sum(batch_val_sparsity) / len(batch_val_sparsity))
            except:
                # when gates stacked
                if len(model.gates.metrics) > 0: 
                    for gm in model.gates.metrics:
                        if gm.name == 'avg_sparsity':
                            sparsity_all_epochs_val[epoch].append(gm.result()) 
                            break 

            callbacks.on_test_batch_end(step, logs={"val_loss": float(batch_loss)})
            callbacks.on_batch_end(step, logs={"val_loss": float(batch_loss)})

        # Aggregated metrics on the val set for this epoch based on our measurements for each step.
        val_mean_loss = sum(
            next(iter(batch[1].items()))[1].shape[0]*batch_loss for batch,batch_loss in zip(val_dataset,batch_loss_all_epochs_val[epoch])
        ) / val_dataset_size
        print(f"\n\t Val loss over epoch: {val_mean_loss}")
        loss_all_epochs_val.append(val_mean_loss)
        
        val_mean_sparsity = sum(
            next(iter(batch[1].items()))[1].shape[0]*batch_sparsity for batch,batch_sparsity in zip(val_dataset,sparsity_all_epochs_val[epoch])
        ) / val_dataset_size
        print(f"\t Avg val sparsity over epoch: {val_mean_sparsity}")
        sparsity_all_epochs_val[epoch] = float(val_mean_sparsity) 
        
        # Change of learning rate if we want to apply a regime change during the training.
        if (
            "train_regime_change" in config.keys()
            and "learning_rate_change" in config["train_regime_change"].keys()
            and "n_epochs_lookback" in config["train_regime_change"].keys()
        ):
            if epoch + 1 > config["train_regime_change"]["n_epochs_lookback"] and all(
                [
                    loss_all_epochs_val[epoch-i]>=loss_all_epochs_val[epoch-config["train_regime_change"]["n_epochs_lookback"]]
                    for i in range(config["train_regime_change"]["n_epochs_lookback"])
                ]
            ):
                optimizer.lr.assign(config["train_regime_change"]["learning_rate_change"] * optimizer.lr.read_value())
                # we assume all gates are of the same type
                if config["gates"][0]["module"] == "TrimmedLassoSimplexProjGate":   
                    try:
                        for gate in model.gates:
                            gate.learning_rate = config["train_regime_change"]["learning_rate_change"] * optimizer.lr.read_value()
                    except:
                        model.gates.learning_rate = config["train_regime_change"]["learning_rate_change"] * optimizer.lr.read_value()                    

        # We display and save val metrics at the end of each epoch.
        with val_summary_writer.as_default():
            tf.summary.scalar("loss", val_mean_loss, step=epoch+1) 
            tf.summary.scalar(f"{config['taskset']['name']} avg_sparsity", val_mean_sparsity, step=epoch+1) 
            val_summary_writer.flush()
        for i,task_metrics in enumerate(task_metrics_list):
            if (
                'display_task_metrics_each_epoch' not in config.keys() 
                or config['display_task_metrics_each_epoch'] == True
            ):
                print(f"\t Validation metrics for Task {config['taskset']['tasks'][i]['name']}")
            for metric in task_metrics:
                metric_value = float(task_metrics[metric].result())
                metrics_all_epochs_val[i][metric].append(metric_value)
                if (
                    'display_task_metrics_each_epoch' not in config.keys() 
                    or config['display_task_metrics_each_epoch'] == True
                ):
                    print(f"\t\t Val {metric} over epoch: {metric_value}")
                # Reset training metrics at the end of each epoch
                try:
                    task_metrics[metric].reset_state()
                except:
                    task_metrics[metric].reset_states()
                with val_summary_writer.as_default():
                    tf.summary.scalar(f"{config['taskset']['tasks'][i]['name']} {metric}", metric_value, step=epoch+1)  
                    val_summary_writer.flush()
            if (
                'display_task_metrics_each_epoch' not in config.keys() 
                or config['display_task_metrics_each_epoch'] == True
            ):
                print("\n")
                    
        for met in model.metrics:
            met.reset_states()
        print("Time taken: %.2fs" % (time.time() - start_time))
        
        # We save weights at the end of the epochs if requested in the arguments.
        if save_weights_every_epoch:
            model.save_weights(weight_saving_path + "/weights/" + "tf_model_" + str(epoch+1) + "." + weight_saving_format)
#             model.save(
#                 weight_saving_path + "/weights/" + "tf_whole_model_" + str(epoch+1) + "." + weight_saving_format
#             )
        
        # We print the actual weight values of the gates if requested.
        try:   
            # when gates not stacked
            with train_summary_writer.as_default():
                for i,gate in enumerate(model.gates):
                    weights = gate.get_weights()
                    if (
                        ('no_weight_printing' not in config.keys())
                        or (not config['no_weight_printing'])
                    ):
                        print(f"Weights of gate for {config['taskset']['tasks'][i]['name']}:")
                        print(weights)
                    try:
                        # need to do a try except because some gates store their weights differently
                        tf.summary.histogram(
                            f"{config['taskset']['tasks'][i]['name']} weights", 
                            weights[0] if isinstance(weights,list) and len(weights) == 1 else weights,
                            step=epoch+1
                        )
                    except:
                        pass

                    try:
                        # need to do a try except because some gates store their weights differently
                        tf.summary.text(
                            f"{config['taskset']['tasks'][i]['name']} weights str", 
                            str(weights[0]) if isinstance(weights,list) and len(weights) == 1 else str(weights),
                            step=epoch+1
                        )
                    except:
                        pass
                train_summary_writer.flush()
            print("\n")
        except:
            # when gates stacked, we do not monitor weights
            pass

        callbacks.on_epoch_end(epoch, logs={
            "loss": train_mean_loss,
            "val_loss": val_mean_loss
        })
                
        # Stop_training check needs to be right after the on_epoch_end method to be able to restore best_weights of the model
        if model.stop_training:
            print(f"\n\t\t - Callback activated: STOPPING AT EPOCH {epoch+1} -\n")         
            break
    
    callbacks.on_train_end(logs=logs)
    
    # At the end of the training, we retrieve train and val quantities measured at best val checkpoint.
    train_sparsity_at_best_val = sparsity_all_epochs_train[-1]
    val_sparsity_at_best_val = sparsity_all_epochs_val[-1]
    best_epoch, _ = min(enumerate(loss_all_epochs_val), key=lambda x: x[1])
    tf.print("Epoch with best val loss: ", best_epoch+1, output_stream=sys.stdout)
    if (
        model.stop_training 
        and best_epoch+1 < config["nb_epochs"]
    ):
        train_sparsity_at_best_val = sparsity_all_epochs_train[best_epoch]
        val_sparsity_at_best_val = sparsity_all_epochs_val[best_epoch]

    if not config["no_weight_saving"]:
        os.mkdir(weight_saving_path + "/weights/trial_" + str(config["trial_nb"]))
        model.save_weights(weight_saving_path + "/weights/trial_" + str(config["trial_nb"]) + "/model_weights")
#         model.save(weight_saving_path + "/weights/" + "tf_model_weights_"+"trial_" + str(config["trial_nb"])+"."+ weight_saving_format)
#         model.save(
#             weight_saving_path + "/weights/" + "tf_whole_model_" + str(epoch+1) + "." + weight_saving_format
#         )
        
    return (
        model, 
        metrics_all_epochs_train, 
        metrics_all_epochs_val, 
        batch_loss_all_epochs_train, 
        batch_loss_all_epochs_val,
        sparsity_all_epochs_train,
        sparsity_all_epochs_val,
        train_sparsity_at_best_val,
        val_sparsity_at_best_val,
        best_epoch+1
    )


# NOT SUPPORTED ANYMORE BECAUSE OF USE OF @tf.function 
def _accumulate_gradients(
    gradients,
    step_gradients,
    num_grad_accumulates
):
    if gradients is None:
        gradients = [_flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += _flat_gradients(g) / num_grad_accumulates       
    return gradients


# NOT SUPPORTED ANYMORE BECAUSE OF USE OF @tf.function 
def _flat_gradients(grads_or_idx_slices):
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices




if __name__ == "__main__":
    # from importlib import reload
    # reload(tf.keras.models)
    main()