from .trimmed_lasso_simplex_proj_gate import TrimmedLassoSimplexProjGate
from .sparse_simplex_proj_gate import SparseSimplexProjGate
from .simplex_proj_gate import SimplexProjGate
from .softmax_gate import SoftmaxGate
from .dselect_k_gate import DSelectKWrapperGate
from .topk_softmax_gate import TopKSoftmaxGate
from .trimmed_lasso_softmax_clipped_gate import TrimmedLassoSoftmaxClippedGate
from .trimmed_lasso_softmax_clipped_quantiles_gate import TrimmedLassoSoftmaxClippedQuantilesGate
from .glam_gate import GLAMGate
from .hash_routing_gate import HashRoutingGate
from .soft_k_trees_ensemble_gate import SoftKTreesEnsembleGate
from .soft_k_trees_ensemble_permuted_gate import SoftKTreesEnsemblePermutedGate
from .soft_k_trees_ensemble_balanced_permuted_gate import SoftKTreesEnsembleBalancedPermutedGate
from .soft_k_trees_ensemble_learn_permuted_gate import SoftKTreesEnsembleLearnPermutedGate
from .tel_gate import TELGate
from .partial_tel_gate import PartialTELGate
from .soft_k_trees_ensemble_shared_alpha_gate import SoftKTreesEnsembleSharedAlphaGate
from .soft_k_trees_ensemble_shared_alpha_permuted_gate import SoftKTreesEnsembleSharedAlphaPermutedGate

from .softmax_gate_stacked import SoftmaxGateStacked
from .sparse_simplex_proj_gate_stacked import SparseSimplexProjGateStacked
from .trimmed_lasso_simplex_proj_gate_stacked import TrimmedLassoSimplexProjGateStacked
from .simplex_proj_gate_stacked import SimplexProjGateStacked
from .topk_softmax_gate_stacked import TopKSoftmaxGateStacked
from .trimmed_lasso_softmax_clipped_gate_stacked import TrimmedLassoSoftmaxClippedGateStacked
from .trimmed_lasso_softmax_clipped_quantiles_gate_stacked import TrimmedLassoSoftmaxClippedQuantilesGateStacked
from .glam_gate_stacked import GLAMGateStacked
from .hash_routing_gate_stacked import HashRoutingGateStacked
from .soft_k_trees_ensemble_gate_stacked import SoftKTreesEnsembleGateStacked
from .soft_k_trees_ensemble_permuted_gate_stacked import SoftKTreesEnsemblePermutedGateStacked
from .soft_k_trees_ensemble_balanced_permuted_gate_stacked import SoftKTreesEnsembleBalancedPermutedGateStacked
from .soft_k_trees_ensemble_learn_permuted_gate_stacked import SoftKTreesEnsembleLearnPermutedGateStacked
from .tel_gate_stacked import TELGateStacked
from .partial_tel_gate_stacked import PartialTELGateStacked
from .soft_k_trees_ensemble_shared_alpha_gate_stacked import SoftKTreesEnsembleSharedAlphaGateStacked
from .soft_k_trees_ensemble_shared_alpha_permuted_gate_stacked import SoftKTreesEnsembleSharedAlphaPermutedGateStacked


GatesMapper = {
    "DSelectKWrapperGate": DSelectKWrapperGate,
    "TrimmedLassoSimplexProjGate": TrimmedLassoSimplexProjGate,
    "SparseSimplexProjGate": SparseSimplexProjGate,
    "SimplexProjGate": SimplexProjGate,
    "SoftmaxGate": SoftmaxGate,
    "TopKSoftmaxGate": TopKSoftmaxGate,
    "TrimmedLassoSoftmaxClippedGate": TrimmedLassoSoftmaxClippedGate,
    "TrimmedLassoSoftmaxClippedQuantilesGate": TrimmedLassoSoftmaxClippedQuantilesGate,
    "GLAMGate": GLAMGate,
    "HashRoutingGate": HashRoutingGate,
    "SoftKTreesEnsembleGate": SoftKTreesEnsembleGate,
    "SoftKTreesEnsemblePermutedGate": SoftKTreesEnsemblePermutedGate,
    "SoftKTreesEnsembleBalancedPermutedGate": SoftKTreesEnsembleBalancedPermutedGate,
    "SoftKTreesEnsembleLearnPermutedGate": SoftKTreesEnsembleLearnPermutedGate,
    "TELGate": TELGate,
    "PartialTELGate": PartialTELGate,
    "SoftKTreesEnsembleSharedAlphaGate": SoftKTreesEnsembleSharedAlphaGate,
    "SoftKTreesEnsembleSharedAlphaPermutedGate": SoftKTreesEnsembleSharedAlphaPermutedGate,
    "SoftmaxGateStacked": SoftmaxGateStacked,
    "SparseSimplexProjGateStacked": SparseSimplexProjGateStacked,
    "TrimmedLassoSimplexProjGateStacked": TrimmedLassoSimplexProjGateStacked,
    "SimplexProjGateStacked": SimplexProjGateStacked,
    "TopKSoftmaxGateStacked": TopKSoftmaxGateStacked,
    "TrimmedLassoSoftmaxClippedGateStacked": TrimmedLassoSoftmaxClippedGateStacked,
    "TrimmedLassoSoftmaxClippedQuantilesGateStacked": TrimmedLassoSoftmaxClippedQuantilesGateStacked,
    "GLAMGateStacked": GLAMGateStacked,
    "HashRoutingGateStacked": HashRoutingGateStacked,
    "SoftKTreesEnsembleGateStacked": SoftKTreesEnsembleGateStacked,
    "SoftKTreesEnsemblePermutedGateStacked": SoftKTreesEnsemblePermutedGateStacked,
    "SoftKTreesEnsembleBalancedPermutedGateStacked": SoftKTreesEnsembleBalancedPermutedGateStacked,
    "SoftKTreesEnsembleLearnPermutedGateStacked": SoftKTreesEnsembleLearnPermutedGateStacked,
    "TELGateStacked": TELGateStacked,
    "PartialTELGateStacked": PartialTELGateStacked,
    "SoftKTreesEnsembleSharedAlphaGateStacked": SoftKTreesEnsembleSharedAlphaGateStacked,
    "SoftKTreesEnsembleSharedAlphaPermutedGateStacked": SoftKTreesEnsembleSharedAlphaPermutedGateStacked,
}