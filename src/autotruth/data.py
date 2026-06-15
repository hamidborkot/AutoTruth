"""
Contrastive pair builder for AutoTruth.
Loads TruthfulQA and constructs (truthful, hallucinated) pairs.
"""

from datasets import load_dataset
from typing import List, Dict
from .config import AutoTruthConfig


def load_truthfulqa(config: AutoTruthConfig):
    """Load TruthfulQA dataset from HuggingFace."""
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split
    )
    return dataset


def build_contrastive_pairs(
    dataset,
    max_samples: int = 200
) -> List[Dict]:
    """
    Build semantically controlled contrastive pairs.

    For each TruthfulQA item, creates:
      - truthful    : "Q: {question}\nA: {correct_answers[0]}"
      - hallucinated: "Q: {question}\nA: {incorrect_answers[0]}"

    Args:
        dataset: HuggingFace TruthfulQA dataset
        max_samples: Maximum number of pairs to build

    Returns:
        List of dicts with keys: question, truthful, hallucinated, category
    """
    pairs = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        q  = item["question"].strip()
        ca = item["correct_answers"]
        ia = item["incorrect_answers"]
        if not ca or not ia:
            continue
        pairs.append({
            "question"    : q,
            "truthful"    : f"Q: {q}\nA: {ca[0].strip()}",
            "hallucinated": f"Q: {q}\nA: {ia[0].strip()}",
            "category"    : item.get("category", "unknown"),
        })
    return pairs
