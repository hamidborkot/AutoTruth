"""
Activation patching engine for AutoTruth.
Implements contrastive activation patching with correct IE direction.
"""

import torch
import numpy as np
import gc
from tqdm import tqdm
from typing import Dict, List, Tuple


def logit_diff(logits_a: torch.Tensor, logits_b: torch.Tensor, top_k: int = 20) -> float:
    """
    Compute mean logit difference between top-k tokens of two distributions.

    Args:
        logits_a: Logits from run A (shape: [1, seq_len, vocab])
        logits_b: Logits from run B (shape: [1, seq_len, vocab])
        top_k: Number of top tokens to average over

    Returns:
        Scalar float: mean(top_k(A)) - mean(top_k(B))
    """
    la = logits_a[0, -1, :]
    lb = logits_b[0, -1, :]
    return (torch.topk(la, top_k)[0].mean() - torch.topk(lb, top_k)[0].mean()).item()


def compute_mean_activations(
    model,
    pairs: List[Dict],
    device: str,
    tokenise_fn,
    hook_filter=None
) -> Dict[str, torch.Tensor]:
    """
    Compute mean activations over all truthful samples.
    Used for principled mean ablation (replaces zero ablation).

    Args:
        model: HookedTransformer model
        pairs: List of contrastive pair dicts
        device: cuda or cpu
        tokenise_fn: Tokenization function
        hook_filter: Lambda for filtering hook names

    Returns:
        Dict mapping hook_name -> mean activation tensor (on device)
    """
    if hook_filter is None:
        hook_filter = lambda n: "hook_z" in n or "hook_mlp_out" in n

    mean_cache  = {}
    count_cache = {}

    model.eval()
    with torch.no_grad():
        for pair in tqdm(pairs, desc="Mean activations"):
            tokens = tokenise_fn(pair["truthful"])
            _, cache = model.run_with_cache(tokens, names_filter=hook_filter)
            for name, act in cache.items():
                val = act[0, -1].detach().cpu()
                if name not in mean_cache:
                    mean_cache[name]  = val.clone()
                    count_cache[name] = 1
                else:
                    mean_cache[name] += val
                    count_cache[name] += 1
            del cache
            gc.collect()

    return {name: (mean_cache[name] / count_cache[name]).to(device)
            for name in mean_cache}


def run_patching_loop(
    model,
    pairs: List[Dict],
    n_layers: int,
    n_heads: int,
    device: str,
    tokenise_fn,
    hook_filter=None
) -> Dict[str, List[float]]:
    """
    Core AutoTruth activation patching loop.

    For each component (layer, head) and (layer, MLP):
      - Run clean forward pass on truthful input
      - Patch in corrupt activation from hallucinated input
      - Measure IE = baseline_LD - patched_LD
      - High positive IE = component is causal for truthfulness

    IMPORTANT: Direction is corrupt->clean patching in CLEAN run.
    This correctly measures how much each component contributes
    to the truthfulness signal.

    Args:
        model: HookedTransformer
        pairs: Contrastive pairs
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        device: cuda or cpu
        tokenise_fn: Tokenization function
        hook_filter: Hook name filter lambda

    Returns:
        Dict mapping component_name -> list of IE values (one per pair)
    """
    if hook_filter is None:
        hook_filter = lambda n: "hook_z" in n or "hook_mlp_out" in n

    scores = {}

    model.eval()
    with torch.no_grad():
        for pair in tqdm(pairs, desc="AutoTruth Patching"):
            tok_clean   = tokenise_fn(pair["truthful"])
            tok_corrupt = tokenise_fn(pair["hallucinated"])

            logits_clean,   cache_clean   = model.run_with_cache(tok_clean,   names_filter=hook_filter)
            logits_corrupt, cache_corrupt = model.run_with_cache(tok_corrupt, names_filter=hook_filter)
            baseline_ie = logit_diff(logits_clean, logits_corrupt)

            # Attention heads
            for layer in range(n_layers):
                hook_name = f"blocks.{layer}.attn.hook_z"
                if hook_name not in cache_corrupt:
                    continue
                for head in range(n_heads):
                    comp_key    = f"L{layer:02d}H{head:02d}"
                    corrupt_act = cache_corrupt[hook_name][0, -1, head].detach().clone()

                    def patch_fn(value, hook, _hn=hook_name, _h=head, _a=corrupt_act):
                        if hook.name == _hn:
                            p = value.clone()
                            p[0, -1, _h] = _a
                            return p
                        return value

                    lp = model.run_with_hooks(tok_clean, fwd_hooks=[(hook_name, patch_fn)])
                    ie = baseline_ie - logit_diff(lp, logits_corrupt)
                    scores.setdefault(comp_key, []).append(ie)

            # MLPs
            for layer in range(n_layers):
                hook_name   = f"blocks.{layer}.hook_mlp_out"
                if hook_name not in cache_corrupt:
                    continue
                comp_key    = f"L{layer:02d}_MLP"
                corrupt_act = cache_corrupt[hook_name][0, -1].detach().clone()

                def mlp_patch(value, hook, _hn=hook_name, _a=corrupt_act):
                    if hook.name == _hn:
                        p = value.clone()
                        p[0, -1] = _a
                        return p
                    return value

                lp = model.run_with_hooks(tok_clean, fwd_hooks=[(hook_name, mlp_patch)])
                ie = baseline_ie - logit_diff(lp, logits_corrupt)
                scores.setdefault(comp_key, []).append(ie)

            del cache_clean, cache_corrupt
            gc.collect()

    return scores
