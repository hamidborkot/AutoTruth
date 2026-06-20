import torch
import numpy as np
from typing import Dict, List, Tuple
from transformer_lens import HookedTransformer


def get_logit_diff(logits: torch.Tensor, tokens_correct: torch.Tensor, tokens_incorrect: torch.Tensor) -> float:
    """Return the logit difference for the last token between correct and incorrect prompts."""
    last_logits = logits[0, -1, :]
    correct_token = tokens_correct[0, -1].item()
    incorrect_token = tokens_incorrect[0, -1].item()
    return (last_logits[correct_token] - last_logits[incorrect_token]).item()


def compute_ie_for_component(
    model: HookedTransformer,
    hook_name: str,
    pairs_tokenized: List[Tuple],
) -> float:
    """Compute mean intervention effect for one component across tokenized pairs."""
    ies = []

    for tc, ti, _ in pairs_tokenized:
        clean_cache = {}

        def save_hook(value, hook):
            clean_cache[hook.name] = value.detach().clone()
            return value

        with torch.no_grad():
            model.run_with_hooks(tc, fwd_hooks=[(hook_name, save_hook)])

        def patch_hook(value, hook):
            return clean_cache[hook.name]

        with torch.no_grad():
            patched_logits = model.run_with_hooks(ti, fwd_hooks=[(hook_name, patch_hook)])
            corrupt_logits = model(ti)

        patched_diff = get_logit_diff(patched_logits, tc, ti)
        corrupt_diff = get_logit_diff(corrupt_logits, tc, ti)
        ies.append(patched_diff - corrupt_diff)

    return float(np.mean(ies))


def discover_circuit(
    model: HookedTransformer,
    pairs_tokenized: List[Tuple],
    K: int = 50,
    verbose: bool = True,
):
    """Discover the top-K circuit components by activation patching IE."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_hooks = []
    for L in range(n_layers):
        all_hooks.append(f"blocks.{L}.hook_mlp_out")
        for H in range(n_heads):
            all_hooks.append(f"blocks.{L}.attn.hook_result")
            break

    ie_scores = {}
    for idx, hook_name in enumerate(all_hooks):
        if verbose and idx % 10 == 0:
            print(f"  Patching {idx}/{len(all_hooks)}: {hook_name}")
        try:
            ie_scores[hook_name] = compute_ie_for_component(model, hook_name, pairs_tokenized)
        except Exception as e:
            print(f"  Skipped {hook_name}: {e}")
            ie_scores[hook_name] = 0.0

    sorted_hooks = sorted(ie_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = [name for name, _ in sorted_hooks[:K]]

    if verbose:
        print(f"\nTop-5 circuit components:")
        for name, ie in sorted_hooks[:5]:
            print(f"  {name}: IE={ie:.4f}")

    return top_k, ie_scores
