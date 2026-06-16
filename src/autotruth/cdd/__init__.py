"""
CDD — Causal Direction Decomposition

Bridges probing and circuit discovery for LLM truthfulness analysis.

Modules:
    cpca   : Contrastive PCA decomposition of circuit components
    rank1  : Rank-1 causal intervention and CRR computation
    probe  : Linear probe training and CPA computation
    domain : Domain-conditional CDD and regime clustering

Key results (GPT-2-XL, TruthfulQA, 200 pairs):
    Mean CPA             : 0.0666  (H1 confirmed: probes != circuits)
    Mean probe accuracy  : 0.9145
    Mean CRR             : 23.101  (100% components above 0.70)
    Mean effective rank  : 57.74
    Regime separation    : Mann-Whitney p = 0.0103
    Context-Driven r     : -0.781 (p = 0.038)
"""

from .cpca import contrastive_pca, collect_component_activations
from .rank1 import rank1_crr
from .probe import train_probe, compute_cpa
from .domain import domain_conditional_cdd, cluster_regimes

__all__ = [
    "contrastive_pca",
    "collect_component_activations",
    "rank1_crr",
    "train_probe",
    "compute_cpa",
    "domain_conditional_cdd",
    "cluster_regimes",
]
