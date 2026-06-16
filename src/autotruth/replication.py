"""
replication.py — Cross-Architecture CPA Replication Runner

Runs the full CDD pipeline on a second model (default: Pythia-1.4B-deduped)
using the same TruthfulQA contrastive pairs protocol as the GPT-2-XL baseline.
Produces results/replication_<model_slug>.json.

Usage:
    python -m src.autotruth.replication --model EleutherAI/pythia-1.4b-deduped
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .cdd.cpca import ContrastivePCA
from .data import build_contrastive_pairs

DEFAULT_MODEL = "EleutherAI/pythia-1.4b-deduped"
N_PAIRS = 200
N_COMPONENTS = 50
SEED = 42
OUTPUT_DIR = Path("results")


def slugify(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").lower()


@torch.no_grad()
def extract_residual_stream(
    model, tokenizer, sentences: list[str], layer_idx: int, device: str
) -> np.ndarray:
    """Extract mean-pooled residual stream at layer_idx for each sentence."""
    reps = []
    model.eval()
    for sent in sentences:
        inputs = tokenizer(
            sent, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden)
        reps.append(hidden[0].mean(0).cpu().float().numpy())
    return np.stack(reps)


def run_replication(
    model_id: str = DEFAULT_MODEL,
    n_pairs: int = N_PAIRS,
    n_components: int = N_COMPONENTS,
    seed: int = SEED,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(device)
    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2

    # Load TruthfulQA
    print("Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    pairs = build_contrastive_pairs(dataset, n_pairs=n_pairs, seed=seed)
    true_sents  = [p["true"]  for p in pairs]
    false_sents = [p["false"] for p in pairs]

    # Extract activations at middle layer
    print(f"Extracting activations at layer {mid_layer}/{n_layers}...")
    X_true  = extract_residual_stream(model, tokenizer, true_sents,  mid_layer, device)
    X_false = extract_residual_stream(model, tokenizer, false_sents, mid_layer, device)

    # --- Probe direction (foreground: truth signal) ---
    X_all = np.concatenate([X_true, X_false], axis=0)
    y_all = np.array([1] * n_pairs + [0] * n_pairs)
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe_acc = cross_val_score(probe, X_all, y_all, cv=5, scoring="accuracy").mean()
    probe.fit(X_all, y_all)
    probe_dir = probe.coef_[0] / (np.linalg.norm(probe.coef_[0]) + 1e-10)

    # --- Circuit direction (cPCA on contrastive pairs) ---
    cpca = ContrastivePCA(n_components=n_components, random_state=seed)
    diff = X_true - X_false
    cpca.fit(diff)
    circuit_dirs = cpca.components_  # (n_components, hidden)

    # --- CPA: cosine alignment between probe dir and each circuit component ---
    cpa_scores = []
    for comp in circuit_dirs:
        comp_norm = comp / (np.linalg.norm(comp) + 1e-10)
        cpa_scores.append(float(np.abs(np.dot(probe_dir, comp_norm))))
    mean_cpa = float(np.mean(cpa_scores))

    result = {
        "model": model_id,
        "n_pairs": n_pairs,
        "n_components": n_components,
        "seed": seed,
        "mid_layer": mid_layer,
        "mean_cpa": round(mean_cpa, 4),
        "mean_probe_accuracy": round(float(probe_acc), 4),
        "h1_confirmed": mean_cpa < 0.15,
        "h1_threshold": 0.15,
        "device": device,
    }

    out_path = OUTPUT_DIR / f"replication_{slugify(model_id)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")
    print(f"Mean CPA = {mean_cpa:.4f} | Probe Acc = {probe_acc:.4f} | H1: {'confirmed' if mean_cpa < 0.15 else 'REJECTED'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n_pairs",    type=int, default=N_PAIRS)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--seed",   type=int, default=SEED)
    args = parser.parse_args()
    run_replication(
        model_id=args.model,
        n_pairs=args.n_pairs,
        n_components=args.n_components,
        seed=args.seed,
    )
