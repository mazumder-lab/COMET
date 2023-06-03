from .dense_classifier import DenseClassifier
from .dense_regressor import DenseRegressor


HeadsMapper = {
    "DenseClassifier": DenseClassifier,
    "DenseRegressor": DenseRegressor,
}