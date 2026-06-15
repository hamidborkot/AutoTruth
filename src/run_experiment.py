"""
AutoTruth — Single-command full pipeline runner.

Usage:
    python src/run_experiment.py --model gpt2-xl --samples 200 --top_k 50 --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from autotruth.config import AutoTruthConfig
from autotruth.data import load_truthfulqa, build_contrastive_pairs
from autotruth.patching import compute_mean_activations, run_patching_loop
from autotruth.scoring import compute_scores, select_circuit
from autotruth.ablation import validate_circuit


def tokenise(model, prompt: str, max_len: int, device: str):
    tokens = model.tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=max_len, padding=False
    )
    return tokens["input_ids"].to(device)


def main(args):
    # Config
    config = AutoTruthConfig(
        model_name=args.model,
        max_samples=args.samples,
        top_k=args.top_k,
        seed=args.seed
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    Path(config.results_dir).mkdir(exist_ok=True)
    Path(config.figures_dir).mkdir(exist_ok=True)

    # Load model
    print(f"Loading {config.model_name}...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(config.model_name, device=config.device)
    model.eval()

    tok_fn = lambda p: tokenise(model, p, config.max_token_length, config.device)

    # Data
    print("Loading TruthfulQA...")
    dataset = load_truthfulqa(config)
    pairs   = build_contrastive_pairs(dataset, config.max_samples)
    print(f"Built {len(pairs)} contrastive pairs")

    # Mean activations
    print("Computing mean activations...")
    mean_cache = compute_mean_activations(model, pairs, config.device, tok_fn)

    # Patching
    print(f"Running patching loop (~5 hours on RTX 5070)...")
    scores = run_patching_loop(
        model, pairs,
        n_layers=model.cfg.n_layers,
        n_heads=model.cfg.n_heads,
        device=config.device,
        tokenise_fn=tok_fn
    )

    # Scoring
    df_all = compute_scores(scores)
    df_circuit, df_neg = select_circuit(df_all, config.top_k, config.threshold_std_multiplier)

    # Save CSVs
    df_all.to_csv(f"{config.results_dir}/autotruth_all_scores_v2.csv", index=False)
    df_circuit.to_csv(f"{config.results_dir}/autotruth_circuit_v2.csv", index=False)
    df_neg.to_csv(f"{config.results_dir}/autotruth_antisuppression_circuit.csv", index=False)
    print(f"Circuit: {len(df_circuit)}/{len(df_all)} components ({len(df_circuit)/len(df_all)*100:.1f}%)")

    # Validation
    print("Running ablation validation...")
    val = validate_circuit(
        model, pairs, df_circuit, df_all, mean_cache, tok_fn,
        config.device, config.ablation_subset_size, config.n_random_seeds
    )

    # Results
    results = {
        "model": config.model_name,
        "n_samples": len(pairs),
        **val,
        "circuit_size": len(df_circuit),
        "total_components": model.cfg.n_layers * (model.cfg.n_heads + 1),
        "sparsity_pct": round(len(df_circuit) / (model.cfg.n_layers*(model.cfg.n_heads+1)) * 100, 2),
        "top_5_circuit": df_circuit["component"].head(5).tolist()
    }

    with open(f"{config.results_dir}/autotruth_final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*52)
    print(f"  Specificity ratio : {val['specificity_ratio']}x")
    print(f"  p-value           : {val['p_value']:.6f}")
    print(f"  Circuit drop      : {val['circuit_drop_pct']}%")
    print(f"  Random drop       : {val['random_drop_pct']}%")
    print(f"  Circuit size      : {len(df_circuit)}/{len(df_all)}")
    print("="*52)
    print(f"Results saved to {config.results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTruth full pipeline")
    parser.add_argument("--model",   default="gpt2-xl",  help="Model name")
    parser.add_argument("--samples", default=200, type=int, help="Number of pairs")
    parser.add_argument("--top_k",   default=50,  type=int, help="Circuit size")
    parser.add_argument("--seed",    default=42,  type=int, help="Random seed")
    main(parser.parse_args())
