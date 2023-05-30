from .softmax import SoftmaxGate
from .dselect_k import DSelectKWrapperGate
from .topk_softmax import TopKSoftmaxGate
from .hash_routing import HashRoutingGate
from .comet import COMETGate

GatesMapper = {
    "DSelectKWrapperGate": DSelectKWrapperGate,
    "SoftmaxGate": SoftmaxGate,
    "TopKSoftmaxGate": TopKSoftmaxGate,
    "HashRoutingGate": HashRoutingGate,
    "COMETGate": COMETGate,
}
