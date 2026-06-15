"""
Centralized configuration for AutoTruth experiments.
Edit ONLY this file to change experimental parameters.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AutoTruthConfig:
    # Model
    model_name: str = "gpt2-xl"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    # Dataset
    dataset_name: str = "truthful_qa"
    dataset_config: str = "generation"
    dataset_split: str = "validation"
    max_samples: int = 200
    max_token_length: int = 128

    # Circuit discovery
    top_k: int = 50
    threshold_std_multiplier: float = 1.0  # score > mean + k*std

    # Ablation
    ablation_subset_size: int = 50
    n_random_seeds: int = 5

    # Reproducibility
    seed: int = 42

    # Output paths
    results_dir: str = "results"
    figures_dir: str = "figures"


# Default config used by all experiments
DEFAULT_CONFIG = AutoTruthConfig()
