"""
creak.py — CREAK Dataset Loader and Contrastive Pair Builder

Loads the CREAK entity-centric claims dataset (amydeng2000/CREAK)
and constructs contrastive pairs: one true claim and one false claim
per Wikipedia entity, used for cross-dataset CPA validation.

Usage:
    from src.autotruth.creak import load_creak_pairs
    pairs = load_creak_pairs(n_pairs=200, seed=42)
"""

import random
from typing import Optional
from datasets import load_dataset


def load_creak_pairs(
    n_pairs: int = 200,
    seed: int = 42,
    split: str = "train",
) -> list[dict]:
    """
    Load CREAK and build contrastive pairs.

    Each pair: {"true": str, "false": str, "entity": str}
    Pair construction: for each entity with both true and false examples,
    sample one true and one false sentence.

    Args:
        n_pairs:  Number of contrastive pairs to return.
        seed:     Random seed for reproducibility.
        split:    Dataset split to use ("train" or "validation").

    Returns:
        List of dicts with keys: true, false, entity.
    """
    print(f"Loading CREAK from amydeng2000/CREAK ...")
    dataset = load_dataset("amydeng2000/CREAK", split=split)

    # Group by entity
    entity_true:  dict[str, list[str]] = {}
    entity_false: dict[str, list[str]] = {}

    for example in dataset:
        entity = example["entity"]
        sentence = example["sentence"]
        label = example["label"]
        if label == "true":
            entity_true.setdefault(entity, []).append(sentence)
        else:
            entity_false.setdefault(entity, []).append(sentence)

    # Keep only entities that have both true and false examples
    entities_with_both = [
        e for e in entity_true
        if e in entity_false
    ]
    print(f"Built {len(entities_with_both)} entities with both true and false claims.")

    rng = random.Random(seed)
    rng.shuffle(entities_with_both)

    pairs = []
    for entity in entities_with_both:
        if len(pairs) >= n_pairs:
            break
        true_sent  = rng.choice(entity_true[entity])
        false_sent = rng.choice(entity_false[entity])
        pairs.append({
            "true":   true_sent,
            "false":  false_sent,
            "entity": entity,
        })

    print(f"Built {len(pairs)} CREAK contrastive pairs from {len(entities_with_both)} unique entities.")
    return pairs


def run_creak_cpa_validation(
    model_id: str = "EleutherAI/pythia-1.4b-deduped",
    n_pairs: int = 200,
    n_components: int = 50,
    seed: int = 42,
    output_path: str = "results/creak_validation.json",
) -> dict:
    """
    Full CPA validation pipeline on CREAK.
    Loads CREAK pairs, extracts activations via Pythia, computes CPA.
    Saves results to output_path.
    """
    import json
    import torch
    import numpy as np
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from .cdd.cpca import ContrastivePCA

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_creak_pairs(n_pairs=n_pairs, seed=seed)
    true_sents  = [p["true"]  for p in pairs]
    false_sents = [p["false"] for p in pairs]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2

    def get_reps(sents):
        reps = []
        with torch.no_grad():
            for sent in sents:
                inputs = tokenizer(
                    sent, return_tensors="pt",
                    truncation=True, max_length=128
                ).to(device)
                out = model(**inputs, output_hidden_states=True)
                h = out.hidden_states[mid_layer][0].mean(0).cpu().float().numpy()
                reps.append(h)
        return np.stack(reps)

    print("Extracting CREAK activations...")
    X_true  = get_reps(true_sents)
    X_false = get_reps(false_sents)

    X_all = np.concatenate([X_true, X_false], axis=0)
    y_all = np.array([1] * n_pairs + [0] * n_pairs)
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe_acc = cross_val_score(probe, X_all, y_all, cv=5, scoring="accuracy").mean()
    probe.fit(X_all, y_all)
    probe_dir = probe.coef_[0] / (np.linalg.norm(probe.coef_[0]) + 1e-10)

    cpca = ContrastivePCA(n_components=n_components, random_state=seed)
    cpca.fit(X_true - X_false)
    cpa_scores = [
        float(np.abs(np.dot(
            probe_dir,
            comp / (np.linalg.norm(comp) + 1e-10)
        )))
        for comp in cpca.components_
    ]
    mean_cpa = float(np.mean(cpa_scores))

    result = {
        "experiment": "CREAK CPA Validation",
        "model": model_id,
        "dataset": "amydeng2000/CREAK",
        "n_pairs": n_pairs,
        "n_components": n_components,
        "seed": seed,
        "mean_cpa": round(mean_cpa, 4),
        "mean_probe_accuracy": round(float(probe_acc), 4),
        "h1_confirmed": mean_cpa < 0.15,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {output_path}")
    print(f"CREAK CPA = {mean_cpa:.4f} | Probe Acc = {probe_acc:.4f}")
    return result
