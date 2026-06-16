"""
cpca.py — Contrastive PCA Decomposition

For each causally-validated circuit component, collect paired activation
differences (truthful - hallucinated) and apply PCA to find the contrastive
principal directions: the subspace that captures truthfulness-specific
variation within each component's output space.

Confirmed result: mean effective rank = 57.74 (GPT-2-XL, TruthfulQA, 200 pairs)
"""

import numpy as np
import torch
import gc
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


def collect_component_activations(model, pairs, df_circuit, tokenise, hook_filter, device):
    """
    Collect paired activation vectors for all circuit components.

    Args:
        model       : TransformerLens HookedTransformer
        pairs       : list of dicts with 'truthful', 'hallucinated', 'category'
        df_circuit  : DataFrame with columns [component, layer, head, type]
        tokenise    : callable, returns token tensor
        hook_filter : callable, hook name filter for run_with_cache
        device      : torch.device

    Returns:
        dict: {comp_key: {'truthful': [np.array, ...], 'hallucinated': [np.array, ...]}}
    """
    component_activations = {}

    model.eval()
    with torch.no_grad():
        for pair in tqdm(pairs, desc="Collecting activations for cPCA"):
            tok_t = tokenise(pair["truthful"])
            tok_h = tokenise(pair["hallucinated"])
            _, cache_t = model.run_with_cache(tok_t, names_filter=hook_filter)
            _, cache_h = model.run_with_cache(tok_h, names_filter=hook_filter)

            for _, row in df_circuit.iterrows():
                comp  = row["component"]
                layer = int(row["layer"])
                if row["type"] == "mlp":
                    hname = f"blocks.{layer}.hook_mlp_out"
                    act_t = cache_t[hname][0, -1].cpu().numpy()
                    act_h = cache_h[hname][0, -1].cpu().numpy()
                else:
                    head  = int(row["head"])
                    hname = f"blocks.{layer}.attn.hook_z"
                    act_t = cache_t[hname][0, -1, head].cpu().numpy()
                    act_h = cache_h[hname][0, -1, head].cpu().numpy()

                if comp not in component_activations:
                    component_activations[comp] = {"truthful": [], "hallucinated": []}
                component_activations[comp]["truthful"].append(act_t)
                component_activations[comp]["hallucinated"].append(act_h)

            del cache_t, cache_h
            gc.collect()

    return component_activations


def contrastive_pca(component_activations, variance_threshold=0.80):
    """
    Apply contrastive PCA to each component's activation differences.

    Args:
        component_activations : output of collect_component_activations
        variance_threshold    : float, fraction of variance to retain (default 0.80)

    Returns:
        causal_directions : dict {comp: np.array of shape (k, d)}
        effective_ranks   : dict {comp: int k}
    """
    causal_directions = {}
    effective_ranks   = {}

    for comp, acts in tqdm(component_activations.items(), desc="cPCA"):
        T = np.array(acts["truthful"])
        H = np.array(acts["hallucinated"])
        delta = T - H
        delta_centered = delta - delta.mean(axis=0)

        pca = PCA()
        pca.fit(delta_centered)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_threshold)) + 1
        k = max(k, 1)

        causal_directions[comp] = pca.components_[:k]   # (k, d)
        effective_ranks[comp]   = k

    ranks = list(effective_ranks.values())
    print(f"cPCA complete")
    print(f"Mean effective rank  : {np.mean(ranks):.2f}")
    print(f"Median effective rank: {np.median(ranks):.1f}")
    print(f"Components with rank=1: {sum(1 for r in ranks if r == 1)}/{len(ranks)}")

    return causal_directions, effective_ranks
