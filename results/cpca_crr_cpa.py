import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


def collect_activations(
    model,
    hook_name: str,
    pairs_tokenized: List[Tuple],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect last-token activations for correct and incorrect prompts."""
    T_acts, H_acts = [], []

    def save_last_token(storage):
        def hook(value, hook):
            storage.append(value[0, -1, :].detach().cpu().float().numpy())
            return value
        return hook

    for tc, ti, _ in pairs_tokenized:
        t_store, h_store = [], []
        with torch.no_grad():
            model.run_with_hooks(tc, fwd_hooks=[(hook_name, save_last_token(t_store))])
            model.run_with_hooks(ti, fwd_hooks=[(hook_name, save_last_token(h_store))])
        if t_store and h_store:
            T_acts.append(t_store[0])
            H_acts.append(h_store[0])

    return np.array(T_acts), np.array(H_acts)


def contrastive_pca(T: np.ndarray, H: np.ndarray, variance_threshold: float = 0.80):
    """Perform contrastive PCA on delta = T - H and return top direction."""
    delta = T - H
    delta_centered = delta - delta.mean(axis=0)
    pca = PCA()
    pca.fit(delta_centered)
    top_direction = pca.components_[0]
    top_direction = top_direction / (np.linalg.norm(top_direction) + 1e-8)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    effective_rank = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return top_direction, effective_rank, pca.explained_variance_ratio_


def compute_crr(
    model,
    hook_name: str,
    top_direction: np.ndarray,
    pairs_tokenized: List[Tuple],
    full_ie: float,
    epsilon: float = 1e-8,
) -> float:
    """Estimate causal recovery ratio using a rank-1 patch."""
    direction_tensor = np.asarray(top_direction, dtype=np.float32)
    rank1_ies = []

    for tc, ti, _ in pairs_tokenized:
        clean_act = {}

        def save_hook(value, hook):
            clean_act[hook.name] = value.detach().clone()
            return value

        with torch.no_grad():
            model.run_with_hooks(tc, fwd_hooks=[(hook_name, save_hook)])

        def rank1_patch_hook(value, hook):
            truthful_act = clean_act[hook.name][0, -1, :]
            corrupt_act = value[0, -1, :]
            proj_coeff = float((truthful_act - corrupt_act).dot(torch.tensor(direction_tensor, device=truthful_act.device)))
            patched = value.clone()
            patched[0, -1, :] += torch.tensor(proj_coeff, device=patched.device) * torch.tensor(direction_tensor, device=patched.device)
            return patched

        with torch.no_grad():
            patched_logits = model.run_with_hooks(ti, fwd_hooks=[(hook_name, rank1_patch_hook)])
            corrupt_logits = model(ti)

        correct_token = tc[0, -1].item()
        incorrect_token = ti[0, -1].item()
        patched_diff = (patched_logits[0, -1, correct_token] - patched_logits[0, -1, incorrect_token]).item()
        corrupt_diff = (corrupt_logits[0, -1, correct_token] - corrupt_logits[0, -1, incorrect_token]).item()
        rank1_ies.append(patched_diff - corrupt_diff)

    ie_rank1 = float(np.mean(rank1_ies))
    return ie_rank1 / (full_ie + epsilon)


def compute_cpa(
    T: np.ndarray,
    H: np.ndarray,
    top_direction: np.ndarray,
    C: float = 100.0,
) -> Tuple[float, float]:
    """Train a probe and compute CPA between probe and causal direction."""
    X = np.vstack([T, H])
    y = np.array([1] * len(T) + [0] * len(H))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_scaled, y)
    probe_acc = clf.score(X_scaled, y)
    probe_dir = clf.coef_[0]
    probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
    cpa = float(np.abs(np.dot(probe_dir, top_direction)))
    return cpa, probe_acc


def random_direction_cpa_baseline(d: int, top_direction: np.ndarray, n_samples: int = 1000) -> float:
    """Estimate CPA baseline for random unit vectors."""
    cpas = []
    for _ in range(n_samples):
        rand_dir = np.random.randn(d)
        rand_dir /= np.linalg.norm(rand_dir)
        cpas.append(abs(np.dot(rand_dir, top_direction)))
    return float(np.mean(cpas))
