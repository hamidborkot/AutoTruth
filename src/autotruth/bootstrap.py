"""
bootstrap.py — Regime Bootstrap Stability Analysis

Bootstraps the three-regime CDD classification across all 18 TruthfulQA
domains. For each domain, resamples 80% of contrastive pairs 1000 times
and measures the fraction of iterations that agree with the full-sample
regime assignment. Produces results/bootstrap_stability.json.

Usage:
    python -m src.autotruth.bootstrap
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from .cdd.domain import classify_regime
from .data import build_contrastive_pairs_by_domain

N_ITERATIONS = 1000
RESAMPLE_FRAC = 0.8
N_COMPONENTS = 50
SEED = 42
STABILITY_THRESHOLD = 0.85
OUTPUT_PATH = Path("results/bootstrap_stability.json")


def bootstrap_domain_stability(
    pairs: list[dict],
    full_regime: str,
    n_iterations: int = N_ITERATIONS,
    resample_frac: float = RESAMPLE_FRAC,
    n_components: int = N_COMPONENTS,
    rng: np.random.Generator = None,
) -> float:
    """
    Compute bootstrap stability for one domain.
    Returns fraction of bootstrap iterations that assign the same regime
    as the full-sample assignment.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(pairs)
    k = max(2, int(n * resample_frac))
    agree = 0
    for _ in range(n_iterations):
        idx = rng.choice(n, size=k, replace=True)
        sample = [pairs[i] for i in idx]
        try:
            regime, dist = classify_regime(sample, n_components=n_components)
            if regime == full_regime:
                agree += 1
        except Exception:
            pass
    return agree / n_iterations


def run_bootstrap(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    domain_pairs = build_contrastive_pairs_by_domain(dataset, seed=seed)

    results = []
    for domain, pairs in tqdm(domain_pairs.items(), desc="Bootstrap"):
        regime, dist = classify_regime(pairs, n_components=N_COMPONENTS)
        stability = bootstrap_domain_stability(
            pairs, regime,
            n_iterations=N_ITERATIONS,
            resample_frac=RESAMPLE_FRAC,
            n_components=N_COMPONENTS,
            rng=rng,
        )
        zone = "STABLE" if stability >= STABILITY_THRESHOLD else "BOUNDARY"
        results.append({
            "domain": domain,
            "regime": regime,
            "mean_dist": round(float(dist), 4),
            "stability": round(float(stability), 4),
            "zone": zone,
        })

    stable_count = sum(1 for r in results if r["stability"] >= STABILITY_THRESHOLD)
    boundary_domains = [r["domain"] for r in results if r["zone"] == "BOUNDARY"]
    core_domains = [r for r in results if abs(r["mean_dist"]) >= 0.15]
    core_mean = float(np.mean([r["stability"] for r in core_domains])) if core_domains else 0.0
    overall_mean = float(np.mean([r["stability"] for r in results]))

    output = {
        "experiment": "Bootstrap Regime Stability",
        "n_iterations": N_ITERATIONS,
        "resample_frac": RESAMPLE_FRAC,
        "n_components": N_COMPONENTS,
        "seed": seed,
        "stability_threshold": STABILITY_THRESHOLD,
        "domains": results,
        "summary": {
            "overall_mean_stability": round(overall_mean, 4),
            "core_domains_mean_stability": round(core_mean, 4),
            "domains_stable": stable_count,
            "total_domains": len(results),
            "boundary_zone_domains": boundary_domains,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Overall mean stability: {overall_mean:.1%}")
    print(f"Core domains mean:      {core_mean:.1%}")
    print(f"Stable domains:         {stable_count}/{len(results)}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_iterations", type=int, default=N_ITERATIONS)
    args = parser.parse_args()
    N_ITERATIONS = args.n_iterations
    run_bootstrap(seed=args.seed)
