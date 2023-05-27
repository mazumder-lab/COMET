import tensorflow as tf
import os
import pandas as pd
import numpy as np
from collections import OrderedDict

from tensorflow.python.keras.backend import dtype


# We adapt the ROOT_PATH to the paths mentioned in the task_config.yml 
# (though we could also change all raw_data_bpath mentioned in the task_config.yml file)
ROOT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../../"
)


def _combine_with_duplicate(root, rel_path):
    rs = root.split("/")
    rps = rel_path.split("/")
    popped = False
    for v in rs:
        if v == rps[0]:
            rps.pop(0)
            popped = True
        elif popped:
            break

    return "/".join(rs+rps)


def open_example_dataset(
    raw_data_bpath,
    task_names,
    taskset_name,
    nb_experts=None,
#     train_batch_size,
#     val_batch_size,
):
    """
    Opens the example dataset for training and validation OR just for test
    args:
    - raw_data_bpath: what truly defines/identifies the shared data pts used across the tasks related to this dataset
    - targets_paths: dict of paths to target vector (or OHE matrix) for each of the tasks we are aiming to solve with our shared features
    (ex: {"task1": "/data/raw/example/targets_task1"})
    - train_batch_size: batch size to use for training (if we want to start a training procedure)
    - val_batch_size: batch size to use for validation or test (if we want to start a validation or testing procedure)
    - trainval: boolean argument to indicate whether we want to train and validate our model or just test it
    returns: 
    - one or two datasets, depending if we want to train and validate the model or just test it
    """
    # print(ROOT_PATH)
    # each task, to then add special operations on targets based on the task we want to achieve (ex: OHE for multiclass classif)
    shared_features_bpath_train = _combine_with_duplicate(
        ROOT_PATH,
        raw_data_bpath + "shared_features/" + "shared_features_train.csv"
    )
    df = pd.read_csv(shared_features_bpath_train, sep=',', header=None)
    shared_features_train = (df.values).astype(np.float32)

    shared_features_bpath_val = _combine_with_duplicate(
        ROOT_PATH,
        raw_data_bpath + "shared_features/" + "shared_features_val.csv"
    )
    df = pd.read_csv(shared_features_bpath_val, sep=',', header=None)
    shared_features_val = (df.values).astype(np.float32)

    shared_features_bpath_test = _combine_with_duplicate(
        ROOT_PATH,
        raw_data_bpath + "shared_features/" + "shared_features_test.csv"
    )
    df = pd.read_csv(shared_features_bpath_test, sep=',', header=None)
    shared_features_test = (df.values).astype(np.float32) 
    
    
    

    # we want to preserve the order of the tasks, so we use an OrderedDict to store our task targets
    targets_train = OrderedDict()     
    targets_val = OrderedDict()
    targets_test = OrderedDict()
    for task_name in task_names:
        targets_path_train = _combine_with_duplicate(
            ROOT_PATH,
            raw_data_bpath + task_name + "_targets/" + task_name + "_targets_train.csv"
        )    
        df = pd.read_csv(targets_path_train, sep=',', header=None)
        task_targets_train = (df.values).astype(np.float32)
        targets_train[task_name] = task_targets_train

        targets_path_val = _combine_with_duplicate(
            ROOT_PATH,
            raw_data_bpath + task_name + "_targets/" + task_name + "_targets_val.csv"
        )    
        df = pd.read_csv(targets_path_val, sep=',', header=None)
        task_targets_val = (df.values).astype(np.float32)
        targets_val[task_name] = task_targets_val

        targets_path_test = _combine_with_duplicate(
            ROOT_PATH,
            raw_data_bpath + task_name + "_targets/"  + task_name + "_targets_test.csv"
        )    
        df = pd.read_csv(targets_path_test, sep=',', header=None)
        task_targets_test = (df.values).astype(np.float32)
        targets_test[task_name] = task_targets_test    


    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input': shared_features_train,
                'indices': tf.one_hot(
                    np.random.randint(
                        low=0,
                        high=nb_experts,
                        size=(shared_features_train.shape[0],)
                    ),
                    nb_experts
                )
            },
            targets_train
        )
    )
#         train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input': shared_features_val,
                'indices': tf.one_hot(
                    np.random.randint(
                        low=0,
                        high=nb_experts,
                        size=(shared_features_val.shape[0],)
                    ),
                    nb_experts
                )
            },
            targets_val
        )
    )
#         val_dataset = val_dataset.shuffle(buffer_size=1024).batch(val_batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                'input': shared_features_test,
                'indices': tf.one_hot(
                    np.random.randint(
                        low=0,
                        high=nb_experts,
                        size=(shared_features_test.shape[0],)
                    ),
                    nb_experts
                )
            },
            targets_test
        )
    )
#         test_dataset = test_dataset.shuffle(buffer_size=1024).batch(val_batch_size)

    return train_dataset, val_dataset, test_dataset
