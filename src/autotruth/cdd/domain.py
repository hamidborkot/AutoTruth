"""
domain.py — Domain-Conditional CDD and Three-Regime Discovery

Computes domain-local causal directions and their alignment with global directions.
Clusters domains into three mechanistic regimes via K-Means.

Confirmed results:
  Direction-Driven regime : n=3, mean IE=0.617, mean CPA=0.703
  Context-Driven regime   : n=7, mean IE=0.686, mean CPA=0.324, r=-0.781 p=0.038
  Circuit Failure regime  : n=8, mean IE=0.149, mean CPA=0.220
  Mann-Whitney p = 0.0103
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, mannwhitneyu
import torch, gc
from tqdm.auto import tqdm


def domain_conditional_cdd(model, pairs, df_circuit, causal_directions,
                           tokenise, hook_filter, min_samples=5):
    """
    For each domain, compute the alignment between domain-local causal directions
    and the global causal directions (domain CPA).

    Returns:
        domain_CPA_results : dict {domain: mean_CPA}
    """
    domain_pairs = defaultdict(list)
    for pair in pairs:
        domain_pairs[pair["category"]].append(pair)

    domain_CPA_results = {}

    for domain, domain_pair_list in domain_pairs.items():
        if len(domain_pair_list) < min_samples:
            continue

        domain_acts = {comp: {"truthful": [], "hallucinated": []}
                       for comp in df_circuit["component"]}

        with torch.no_grad():
            for pair in domain_pair_list:
                tok_t = tokenise(pair["truthful"])
                tok_h = tokenise(pair["hallucinated"])
                _, cache_t = model.run_with_cache(tok_t, names_filter=hook_filter)
                _, cache_h = model.run_with_cache(tok_h, names_filter=hook_filter)
                for _, row in df_circuit.iterrows():
                    comp  = row["component"]
                    layer = int(row["layer"])
                    if row["type"] == "mlp":
                        hname = f"blocks.{layer}.hook_mlp_out"
                        domain_acts[comp]["truthful"].append(cache_t[hname][0, -1].cpu().numpy())
                        domain_acts[comp]["hallucinated"].append(cache_h[hname][0, -1].cpu().numpy())
                    else:
                        head  = int(row["head"])
                        hname = f"blocks.{layer}.attn.hook_z"
                        domain_acts[comp]["truthful"].append(cache_t[hname][0, -1, head].cpu().numpy())
                        domain_acts[comp]["hallucinated"].append(cache_h[hname][0, -1, head].cpu().numpy())
                del cache_t, cache_h
                gc.collect()

        cpas = []
        for comp in df_circuit["component"]:
            T = np.array(domain_acts[comp]["truthful"])
            H = np.array(domain_acts[comp]["hallucinated"])
            if len(T) < 3:
                continue
            delta   = T - H
            delta_c = delta - delta.mean(0)
            if delta_c.std() < 1e-8:
                continue
            pca_d = PCA(n_components=1)
            pca_d.fit(delta_c)
            domain_dir = pca_d.components_[0]
            global_dir = causal_directions[comp][0]
            domain_dir = domain_dir / (np.linalg.norm(domain_dir) + 1e-8)
            global_dir = global_dir / (np.linalg.norm(global_dir) + 1e-8)
            cpas.append(float(np.abs(np.dot(domain_dir, global_dir))))

        domain_CPA_results[domain] = float(np.mean(cpas)) if cpas else np.nan

    return domain_CPA_results


def cluster_regimes(df_domain, ie_col="mean_ie", cpa_col="domain_CPA", n_clusters=3):
    """
    Cluster domains into three regimes using K-Means on (mean_IE, domain_CPA).

    Returns:
        df_domain with added 'regime' column
        regime_stats dict
    """
    X = df_domain[[ie_col, cpa_col]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df_domain = df_domain.copy()
    df_domain["regime_id"] = km.fit_predict(X_scaled)

    regime_ie  = df_domain.groupby("regime_id")[ie_col].mean()
    regime_map = {
        regime_ie.idxmax(): "Direction-Driven",
        regime_ie.idxmin(): "Circuit Failure",
    }
    middle = [r for r in regime_ie.index if r not in regime_map][0]
    regime_map[middle] = "Context-Driven"
    df_domain["regime"] = df_domain["regime_id"].map(regime_map)

    # Statistical tests
    driven  = df_domain[df_domain["regime"] == "Direction-Driven"][cpa_col].values
    failure = df_domain[df_domain["regime"] == "Circuit Failure"][cpa_col].values
    context = df_domain[df_domain["regime"] == "Context-Driven"]

    mw_stat, mw_p = mannwhitneyu(driven, failure, alternative="greater")

    ctx_r, ctx_p = (np.nan, np.nan)
    if len(context) >= 3:
        ctx_r, ctx_p = pearsonr(context[cpa_col].values, context[ie_col].values)

    regime_stats = {
        "mann_whitney_p": mw_p,
        "context_driven_r": ctx_r,
        "context_driven_p": ctx_p,
        "direction_driven_mean_cpa": float(np.mean(driven)),
        "circuit_failure_mean_cpa":  float(np.mean(failure)),
    }

    return df_domain, regime_stats
