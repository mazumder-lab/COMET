from .dense_expert import DenseExpert
from .dense_expert_dropout import DenseExpertDropout
from .dense_expert_sum import DenseExpertSum
from .conv_expert import ConvExpert

from .dense_expert_dropout_stacked import DenseExpertDropoutStacked
from .dense_expert_sum_stacked import DenseExpertSumStacked

from .linear_expert import LinearExpert
from .linear_expert_stacked import LinearExpertStacked

ExpertsMapper = {
    "DenseExpert": DenseExpert,
    "DenseExpertDropout": DenseExpertDropout,
    "DenseExpertSum": DenseExpertSum,
    "ConvExpert": ConvExpert,
    "DenseExpertDropoutStacked": DenseExpertDropoutStacked,
    "DenseExpertSumStacked": DenseExpertSumStacked,
    "LinearExpert": LinearExpert,
    "LinearExpertStacked": LinearExpertStacked
}