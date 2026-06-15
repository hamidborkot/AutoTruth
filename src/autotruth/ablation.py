"""
Mean ablation validator for AutoTruth.
Provides statistically rigorous causality validation.
"""

import torch
import numpy as np
import scipy.stats as stats
from typing import Set, Dict, Tuple
from .patching import logit_diff


def run_mean_ablation(
    model,
    pairs_subset: list,
    component_set: Set[str],
    mean_cache: Dict[str, torch.Tensor],
    tokenise_fn,
    device: str
) -> np.ndarray:
    """
    Mean ablation: replace circuit component activations with dataset mean.
    Returns array of logit differences after ablation.

    IMPORTANT: Uses mean activation (not zero) to stay in-distribution.

    Args:
        model: HookedTransformer
        pairs_subset: Subset of contrastive pairs for ablation
        component_set: Set of component names to ablate
        mean_cache: Dataset mean activations (from compute_mean_activations)
        tokenise_fn: Tokenization function
        device: cuda or cpu

    Returns:
        np.ndarray of logit differences after ablation
    """
    result_lds = []

    with torch.no_grad():
        for pair in pairs_subset:
            tok_clean   = tokenise_fn(pair["truthful"])
            tok_corrupt = tokenise_fn(pair["hallucinated"])

            hooks = []
            for comp in component_set:
                if "_MLP" in comp:
                    layer = int(comp[1:3])
                    hname = f"blocks.{layer}.hook_mlp_out"
                    if hname not in mean_cache:
                        continue
                    mv = mean_cache[hname].clone()
                    def make_mlp_hook(hn, mv_):
                        def fn(val, hook):
                            if hook.name == hn:
                                v = val.clone(); v[0, -1] = mv_; return v
                            return val
                        return fn
                    hooks.append((hname, make_mlp_hook(hname, mv)))
                else:
                    layer = int(comp[1:3]); head = int(comp[4:6])
                    hname = f"blocks.{layer}.attn.hook_z"
                    if hname not in mean_cache:
                        continue
                    mv = mean_cache[hname]
                    hm = mv[head] if mv.dim() >= 1 and mv.shape[0] > head else mv
                    def make_attn_hook(hn, h_, hm_):
                        def fn(val, hook):
                            if hook.name == hn:
                                v = val.clone(); v[0, -1, h_] = hm_; return v
                            return val
                        return fn
                    hooks.append((hname, make_attn_hook(hname, head, hm)))

            logits_abl = model.run_with_hooks(tok_clean, fwd_hooks=hooks)
            logits_cor, _ = model.run_with_cache(tok_corrupt, names_filter=lambda n: False)
            result_lds.append(logit_diff(logits_abl, logits_cor))

    return np.array(result_lds)


def validate_circuit(
    model,
    pairs: list,
    df_circuit,
    df_all,
    mean_cache: Dict,
    tokenise_fn,
    device: str,
    ablation_size: int = 50,
    n_random_seeds: int = 5
) -> Dict:
    """
    Full circuit validation: circuit ablation vs. random ablation with t-test.

    Args:
        model: HookedTransformer
        pairs: All contrastive pairs
        df_circuit: Promoting circuit DataFrame
        df_all: All scored components DataFrame
        mean_cache: Dataset mean activations
        tokenise_fn: Tokenization function
        device: cuda or cpu
        ablation_size: Number of pairs for ablation (default 50)
        n_random_seeds: Random seeds for baseline (default 5)

    Returns:
        Dict with all validation metrics
    """
    abl_pairs = pairs[:ablation_size]

    # Baseline
    baseline_lds = []
    with torch.no_grad():
        for pair in abl_pairs:
            lt, _ = model.run_with_cache(tokenise_fn(pair["truthful"]),   names_filter=lambda n: False)
            lh, _ = model.run_with_cache(tokenise_fn(pair["hallucinated"]),names_filter=lambda n: False)
            baseline_lds.append(logit_diff(lt, lh))
    baseline_arr = np.array(baseline_lds)

    # Circuit ablation
    circuit_arr = run_mean_ablation(
        model, abl_pairs, set(df_circuit["component"]),
        mean_cache, tokenise_fn, device
    )

    # Random ablation (averaged over n seeds)
    random_arrs = []
    for seed in range(n_random_seeds):
        rand_comps = set(df_all.sample(len(df_circuit), random_state=seed)["component"])
        random_arrs.append(run_mean_ablation(
            model, abl_pairs, rand_comps, mean_cache, tokenise_fn, device
        ))
    random_arr = np.mean(random_arrs, axis=0)

    abs_base          = abs(baseline_arr.mean()) + 1e-6
    circuit_drop_pct  = (baseline_arr.mean() - circuit_arr.mean()) / abs_base * 100
    random_drop_pct   = (baseline_arr.mean() - random_arr.mean())  / abs_base * 100
    specificity       = abs(circuit_drop_pct) / max(abs(random_drop_pct), 1e-6)
    t_stat, p_val     = stats.ttest_rel(baseline_arr, circuit_arr)

    return {
        "baseline_mean"    : float(baseline_arr.mean()),
        "baseline_std"     : float(baseline_arr.std()),
        "circuit_abl_mean" : float(circuit_arr.mean()),
        "circuit_abl_std"  : float(circuit_arr.std()),
        "random_abl_mean"  : float(random_arr.mean()),
        "circuit_drop_pct" : round(circuit_drop_pct, 1),
        "random_drop_pct"  : round(random_drop_pct, 1),
        "specificity_ratio": round(specificity, 1),
        "t_statistic"      : round(float(t_stat), 4),
        "p_value"          : float(p_val),
        "significant"      : bool(p_val < 0.05)
    }
