from .hash_routing import HashRoutingGate
from .topk_softmax import TopKSoftmaxGate
from .dselect_k import DSelectKWrapperGate
from .comet import COMETGate

GatesMapper = {
    "HashRoutingGate": HashRoutingGate,
    "TopKSoftmaxGate": TopKSoftmaxGate,
    "DSelectKWrapperGate": DSelectKWrapperGate,
    "COMETGate": COMETGate,
}
