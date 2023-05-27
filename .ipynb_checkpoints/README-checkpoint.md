# COMET

## Overview and preliminary steps
There are __3 commands you can run__:
- One to train a single model (```train_tasks.py```)
- One to test a pretrained model (```test_tasks.py```)
- One to perform hyperparameter search (using the TPE suggest algorithm from hyperopt) (```main.py```)


BUT before running any of these commands, __you need to define__:
- The different __tasks y^(1), y^(2),…,y^(K) we are aiming to solve__ for a given set of data points X (and how we’d like to evaluate these tasks, where we’ve stored the shared data points x and their various labels y^(1), y^(2),…,y^(K), etc…)
- The __config of our model__ (where we detail its MoE architecture and all the hyperparameters that come with it, such as a ```suggest_categorical``` sampling on the sparsity level of our gates)
- The __config of our training/testing/hyperparameter search procedure__ (where we detail hyperparameters such the weight we’d like to assign to each of the K tasks, the type of optimizer we’d like to use, its learning rate, the train batch size, the callbacks, the number of epochs,…)

Thus, __only the model config and the train config files can contain hyperparameters to tune__; not the task configs. This is why each specific model architecture needs to be described in its own ```model_config.json``` file. Same goes for each different training procedure, whose hyperparameters need to be defined in a ```train_config.json``` file. However, in contrast, all the different tasks supported are characterized in a single yaml file named ```task_configs.yml```.

**Note:** When running hyperparameter search using a specific ```model_config.json``` file along with a specific ```train_config.json``` file, to declare the values to try out of a given model hyperparameter named ```my_hyperparameter```, the only thing you have to add to your ```model_config.json```file is:
```
"my_hyperparameter": {
	"optuna_type": "suggest_categorical" 		# or "suggest_uniform"
	"values": ["value1", "value2", "value3"]		# or interval of floats
}
```
for instance. However, using this model config file to launch a simple training procedure (without hyperparameter tuning) would raise an error, as a fixed value for "my_hyperparameter" is not defined here. The ```model_config.json``` file would need to be modified (or a new one created).



## Running commands
Here are the __3 available commands__ in detail:
```
python3 train_tasks.py 
--from_pretrained optional_path_to_pretrained_weights_of_our_architecture
--model_config path_to_specific_model_config_file 
--train_config path_to_specific_train_config_file 
--task_config path_to_all_task_configs 
--results_location bpath_of_folder_where_to_store_results 
--experiment_name optional_name_of_folder_to_create_with_results
--seed optional_int_seed_for_reproducibility
--save_weights_every_epoch
--use_MoE_stacked
```

```
python3 test_tasks.py 
--from_pretrained optional_path_to_pretrained_weights_of_our_architecture
--model_config path_to_specific_model_config_file 
--train_config path_to_specific_train_config_file_containing_val_hyperparams
--task_config path_to_all_task_configs 
--results_location bpath_of_folder_where_to_store_results 
--experiment_name optional_name_of_folder_to_create_with_results
--ground_truth_weights_location path_to_gt_weights_for_support_recovery
--use_test_dataset true_or_false
--perform_weight_recovery_analysis 
--use_MoE_stacked
```

```
python3 main.py 
--from_pretrained optional_path_to_pretrained_weights_of_our_architecture
--model_config path_to_specific_model_config_file_with_hyperopt_fields 
--train_config path_to_specific_train_config_file_with_hyperopt_fields 
--task_config path_to_all_task_configs 
--results_location bpath_of_folder_where_to_store_results 
--experiment_name optional_name_of_folder_to_create_with_results
--seed optional_int_seed_for_reproducibility
--max_hyperparameter_evals optional_int_nb_of_trials_with_different_combinations_of_hyperparameters
--no_weight_saving
--perform_weight_recovery_analysis
--use_MoE_stacked
```


## How to's
- Add a new module to the architecture:
> 1. navigate to the ```./model``` folder
> 2. create your custom module in the subfolder corresponding to the type of module you wish to implement (```shared_bottom```, ```expert```, ```gate```, or ```task_specific_head```)
> 3. code your module in it and test it
> 4. add it to the mapper in the ```__init__.py``` file located in the subfolder you are in
> 5. modify your model config by mentioning the class name of your new custom module.

- Add a new taskset to the collection of taskset:
> 1. if your taskset is already in the format (X, [y^(1), y^(2),…,y^(K)]), create a folder in ```./data/raw/``` of the name ```my_new_taskset_name``` for example (it has to be the name of your taskset). Otherwise, create a script in ```./data/utils/data_generators``` to adapt it.
> 2. within ```./data/raw/my_new_taskset_name/```, create one subfolder ```shared_features``` and another ```my_new_taskset_name_targets```. The former subfolder should contain X_train, X_val, and X_test. The latter should contain an aggregate in .csv format of the targets y^(1), y^(2),…,y^(K) of the K tasks for the train, val, and test datasets (**respectively named ```my_new_taskset_name_targets_train.csv```, ```my_new_taskset_name_targets_val.csv```, and ```my_new_taskset_name_targets_test.csv```**)
> 3. create a custom dataset opener (drawing inspiration from/re-using the existing ones) for this dataset in ```./data/dataset_openers/```; it should return TF dataloaders for the train and val or just test datasets, with each time an OrderedDict containing the task labels 
> 4. add the custom dataset opener you just created to the mapper in the ```__init__.py``` file located in the subfolder you are in.
> 5. add information about your new taskset to the file ```./config/task_config/task_configs.yml```: taskset id, taskset name, task names, path to the (raw) data, loss to use for each task, metrics to use for each task, val quantities to monitor for hyperparameter search,...
> 6. create a train config file for it; make sure to mention the right taskset_id in it!
