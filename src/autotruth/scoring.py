"""
Circuit scoring and ranking for AutoTruth.
Computes AutoTruth Score = mean_IE x Consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def compute_scores(scores: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Compute AutoTruth Score for each component.

    Score = mean_IE x Consistency
      where Consistency = fraction of pairs where IE > 0

    Args:
        scores: Dict from patching loop (component -> list of IE values)

    Returns:
        DataFrame sorted by score descending
    """
    rows = []
    for comp, ie_list in scores.items():
        arr         = np.array(ie_list)
        mean_ie     = float(arr.mean())
        consistency = float((arr > 0).mean())
        score       = mean_ie * consistency

        if "_MLP" in comp:
            layer = int(comp[1:3]); head = -1; ctype = "mlp"
        else:
            layer = int(comp[1:3]); head = int(comp[4:6]); ctype = "attn"

        rows.append({
            "component": comp, "type": ctype,
            "layer": layer, "head": head,
            "mean_ie": mean_ie, "consistency": consistency, "score": score
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def select_circuit(
    df_all: pd.DataFrame,
    top_k: int = 50,
    std_multiplier: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select promoting and suppressing circuits using statistical threshold.

    Circuit = components with score > mean + std_multiplier * std
    Anti-circuit = components with score < mean - std_multiplier * std

    Falls back to top_k if threshold gives fewer than 10 components.

    Args:
        df_all: Full scored DataFrame
        top_k: Fallback circuit size
        std_multiplier: Threshold strictness (default 1.0 = 1 std above mean)

    Returns:
        Tuple of (promoting_circuit_df, suppressing_circuit_df)
    """
    mu    = df_all["score"].mean()
    sigma = df_all["score"].std()

    promoting   = df_all[df_all["score"] >= mu + std_multiplier * sigma].copy()
    suppressing = df_all[df_all["score"] <= mu - std_multiplier * sigma].copy()

    if len(promoting) < 10:
        promoting = df_all.head(top_k).copy()

    return promoting, suppressing
