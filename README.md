# COMET

This is the offical repo of the KDD 2023 paper: COMET: Learning Cardinality Constrained Mixture of Experts
with Trees and Local Search

The repository contains code to run Mixture of Expert models with different gates e.g., COMET, Topk, DSelect-k, Hash routing etc. on MovieLens. 

For distillation experiments with COMET-BERT (MoE-based variant of BERT with COMET gates), please refer to this COMET-BERT repo: https://github.com/mazumder-lab/COMET-BERT 



## Requirements
The codebase has been tested with Python 3.8 and the following packages
```
numpy==1.21.6
scikit_learn==1.0.2
tensorflow==2.4.1
```


## Running commands
Here is the available command to perform hyperparameter search (using the random search algorithm from hyperopt):

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
```


## Citing COMET

If you find COMET useful in your research, please consider citing the following paper.

@inproceedings{Ibrahim2023,
author = {Ibrahim, Shibal and Chen, Wenyu and Hazimeh, Hussein and Ponomareva, Natalia and Zhao, Zhe and Mazumder, Rahul},
title = {COMET: Learning Cardinality Constrained Mixture of Experts with Trees and Local Search},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539412},
doi = {10.1145/3534678.3539412},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {666–675},
numpages = {10},
keywords = {negative binomial regression, differentiable trees, flexible loss, zero-inflation models, tree ensemble learning, multi-task learning},
location = {Washington DC, USA},
series = {KDD '22}
}

```python

```
