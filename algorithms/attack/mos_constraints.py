"""
MOS-Attack Constraint System
Radial and Sign constraints for stealth optimization.
"""

import torch
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Type Annotations
# ============================================================================
LayerDims = List[Tuple[str, int, int]]

# ============================================================================
# Constraint Plugin Base
# ============================================================================
class ConstraintPlugin:
    """Base class for constraint plugins"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.threshold: Optional[torch.Tensor] = None

    def fit(self, benign_updates: torch.Tensor, benign_mean: torch.Tensor,
            context: Dict) -> None:
        raise NotImplementedError

    def loss(self, population: torch.Tensor, benign_mean: torch.Tensor,
             context: Dict) -> torch.Tensor:
        raise NotImplementedError

    def score(self, population: torch.Tensor, benign_mean: torch.Tensor,
              context: Dict, eps: float = 1e-12) -> torch.Tensor:
        loss_val = self.loss(population, benign_mean, context)
        ratio = loss_val / (self.threshold + eps)
        return self.score_from_ratio(ratio)

    def score_from_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        """Compute score from pre-computed ratio. Override for custom scoring."""
        return 1.0 / (1.0 + ratio)

# ============================================================================
# Radial Constraint (Distance from benign mean)
# ============================================================================
class RadialConstraint(ConstraintPlugin):
    """Radial distance constraint (L2 norm from benign mean)"""

    def __init__(self, weight: float = 1.0, quantile: float = 0.95):
        super().__init__(name='radial', weight=weight)
        self.quantile = quantile

    def fit(self, benign_updates: torch.Tensor, benign_mean: torch.Tensor,
            context: Dict) -> None:
        dists = torch.norm(benign_updates - benign_mean, dim=1)
        self.threshold = torch.quantile(dists, q=self.quantile)
        self.threshold = torch.clamp(self.threshold, min=1e-6)

    def loss(self, population: torch.Tensor, benign_mean: torch.Tensor,
             context: Dict) -> torch.Tensor:
        """Compute L2 distance from benign mean"""
        return torch.norm(population - benign_mean, dim=1)

# ============================================================================
# Sign Constraint (Layer-normalized sign violation)
# ============================================================================
class SignConstraint(ConstraintPlugin):
    """Layer-normalized sign constraint"""

    def __init__(self, weight: float = 0.5, quantile: float = 0.95,
                 layer_reduce: str = 'quantile', layer_quantile: float = 0.9):
        super().__init__(name='sign', weight=weight)
        self.quantile = quantile
        self.layer_reduce = layer_reduce
        self.layer_quantile = layer_quantile

    def fit(self, benign_updates: torch.Tensor, benign_mean: torch.Tensor,
            context: Dict) -> None:
        layer_dims = context['layer_dims']
        benign_losses = self._compute_layer_losses(
            benign_updates, benign_mean, layer_dims
        )
        self.threshold = torch.quantile(benign_losses, q=self.quantile)
        self.threshold = torch.clamp(self.threshold, min=1e-6)

    def _compute_layer_losses(self, population: torch.Tensor,
                             benign_mean: torch.Tensor,
                             layer_dims: LayerDims) -> torch.Tensor:
        layer_losses = []

        for layer_name, start_idx, end_idx in layer_dims:
            layer_pop = population[:, start_idx:end_idx]
            layer_mean = benign_mean[start_idx:end_idx]

            # Sign violation: negative dot product with mean's sign
            sign_violation = -layer_pop * torch.sign(layer_mean).unsqueeze(0)

            # ReLU to keep only violations, then normalize by layer norm
            violation_norm = torch.norm(torch.relu(sign_violation), dim=1)
            reference_norm = torch.norm(layer_mean) + 1e-12
            layer_loss = violation_norm / reference_norm

            layer_losses.append(layer_loss)

        layer_losses = torch.stack(layer_losses, dim=0)  # (L, P)

        # Aggregate across layers
        if self.layer_reduce == 'max':
            return layer_losses.max(dim=0)[0]
        elif self.layer_reduce == 'mean':
            return layer_losses.mean(dim=0)
        elif self.layer_reduce == 'quantile':
            return torch.quantile(layer_losses, q=self.layer_quantile, dim=0)
        else:
            return layer_losses.max(dim=0)[0]

    def loss(self, population: torch.Tensor, benign_mean: torch.Tensor,
             context: Dict) -> torch.Tensor:
        """Compute sign constraint loss"""
        layer_dims = context['layer_dims']
        return self._compute_layer_losses(population, benign_mean, layer_dims)

# ============================================================================
# Attack Budget Projection
# ============================================================================
def project_to_attack_budget(
    population: torch.Tensor,
    benign_mean: torch.Tensor,
    max_dev_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project population to attack budget

    Returns:
        projected: Projected population
        clipped_mask: Boolean mask of clipped individuals
        pre_norms: Pre-projection norms
    """
    centered = population - benign_mean
    norms = torch.norm(centered, dim=1, keepdim=True)

    clipped_mask = norms.squeeze(1) > max_dev_threshold
    scales = torch.clamp(max_dev_threshold / (norms + 1e-12), max=1.0)
    projected = benign_mean + centered * scales

    return projected, clipped_mask, norms.squeeze(1)

# ============================================================================
# Constraint Scoring and CV Computation
# ============================================================================
def compute_constraint_pass_score(
    population: torch.Tensor,
    benign_mean: torch.Tensor,
    constraints: List[ConstraintPlugin],
    context: Dict,
    eps: float = 1e-12,
    pre_computed_losses: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute weighted constraint pass score

    Returns:
        total_score: Weighted average score (higher = more stealthy)
        scores_dict: Individual constraint scores
    """
    P = population.shape[0]
    device = population.device

    scores_dict = {}
    total_score = torch.zeros(P, device=device)
    total_weight = 0.0

    for constraint in constraints:
        if pre_computed_losses is not None and constraint.name in pre_computed_losses:
            loss_val = pre_computed_losses[constraint.name]
        else:
            loss_val = constraint.loss(population, benign_mean, context)

        ratio = loss_val / (constraint.threshold + eps)
        score_val = constraint.score_from_ratio(ratio)

        scores_dict[constraint.name] = score_val
        total_score += constraint.weight * score_val
        total_weight += constraint.weight

    total_score = total_score / (total_weight + eps)
    return total_score, scores_dict

# ============================================================================
# Dual-Objective Computation
# ============================================================================
def compute_dual_objectives(
    population: torch.Tensor,
    benign_mean: torch.Tensor,
    constraints: List[ConstraintPlugin],
    attack_guidance: torch.Tensor,
    context: Dict
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute dual objectives for NSGA-II

    Objective 1: max R(x) - constraint pass score
    Objective 2: max A(x) - destructiveness (alignment with guidance)

    Returns:
        objectives: (2, P) matrix in minimization form
        constraint_pass_scores: (P,) total constraint scores
        scores_dict: Per-constraint scores
        total_cv: (P,) constraint violations
        ratios_dict: Per-constraint loss/threshold ratios
    """
    eps = 1e-12

    # Compute all constraint losses once
    losses_dict = {}
    for constraint in constraints:
        losses_dict[constraint.name] = constraint.loss(population, benign_mean, context)

    # Compute constraint pass scores using pre-computed losses
    constraint_pass_scores, scores_dict = compute_constraint_pass_score(
        population, benign_mean, constraints, context, eps, pre_computed_losses=losses_dict
    )

    # Compute destructiveness (alignment with attack guidance)
    centered = population - benign_mean
    destructiveness = centered @ attack_guidance

    # Compute CV and ratios inline
    total_cv = torch.zeros(population.shape[0], device=population.device)
    ratios_dict = {}
    for constraint in constraints:
        loss_val = losses_dict[constraint.name]
        ratio = loss_val / (constraint.threshold + eps)
        ratios_dict[constraint.name] = ratio
        violation = torch.relu(ratio - 1.0)
        total_cv += constraint.weight * violation

    # Convert to minimization form
    obj1 = -constraint_pass_scores
    obj2 = -destructiveness
    objectives = torch.stack([obj1, obj2], dim=0)

    return objectives, constraint_pass_scores, scores_dict, total_cv, ratios_dict
