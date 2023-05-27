# Define dataset mapper here.
from .example_dataset_opener import open_example_dataset
from .movielens_dataset_opener import open_movielens_dataset
from .large_scale_synthetic_regressions_dataset_opener import open_large_scale_synthetic_regressions_dataset
from .image_multiclass_classifications_dataset_opener import open_image_multiclass_classifications_dataset
from .image_rgb_classifications_dataset_opener import open_image_rgb_classifications_dataset
from .bert_tokenized_text_multilabel_classification_dataset_opener import open_bert_tokenized_text_multilabel_classification_dataset
from .tensorflow_datasets_opener import open_tensorflow_dataset
from .mulan_datasets_opener import open_mulan_dataset

# maps the name of each taskset with its opener
DatasetOpenerMapper = {
    "synthetic_regressions": open_example_dataset,
    "movielens": open_movielens_dataset,
    "books": open_movielens_dataset,
    "jester": open_movielens_dataset,
    "synthetic_regressions_s": open_example_dataset,
    "synthetic_regressions_frozen_experts": open_example_dataset,
    "movielens_s": open_movielens_dataset,
    "large_scale_synthetic_regressions": open_large_scale_synthetic_regressions_dataset,
    "large_scale_synthetic_regressions_frozen_experts": open_large_scale_synthetic_regressions_dataset,
    "instance_specific_large_scale_synthetic_regressions": open_large_scale_synthetic_regressions_dataset,
    "instance_specific_large_scale_synthetic_regressions_frozen_experts": open_large_scale_synthetic_regressions_dataset,
    "multi_mnist": open_image_multiclass_classifications_dataset,
    "multi_fashion_mnist": open_image_multiclass_classifications_dataset,
    "go_emotions": open_bert_tokenized_text_multilabel_classification_dataset,
    "celebA": open_tensorflow_dataset, #open_image_rgb_classifications_dataset
    "mixture_of_digits": open_tensorflow_dataset, #open_image_rgb_classifications_dataset
    "rf1": open_mulan_dataset,
    "scm1d": open_mulan_dataset
}