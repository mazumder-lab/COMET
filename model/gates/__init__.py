from .hash_routing import HashRoutingGate
from .topk import TopkGate
from .dselect_k import DSelectKWrapperGate
from .comet import COMETGate

GatesMapper = {
    "HashRoutingGate": HashRoutingGate,
    "TopkGate": TopkGate,
    "DSelectKWrapperGate": DSelectKWrapperGate,
    "COMETGate": COMETGate,
}
