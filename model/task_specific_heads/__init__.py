from .empty_head import EmptyHead
from .dense_classifier import DenseClassifier
from .dense_multiclass_classifier import DenseMulticlassClassifier
from .dense_regressor import DenseRegressor
from .linear_regressor import LinearRegressor


HeadsMapper = {
    "DenseClassifier": DenseClassifier,
    "DenseMulticlassClassifier": DenseMulticlassClassifier,
    "DenseRegressor": DenseRegressor,
    "EmptyHead": EmptyHead,
    "LinearRegressor": LinearRegressor
}