# AutoTruth Results Index

> **AAAI 2027 Submission** | Paper deadline: July 21 2026  
> Single source of truth for paper numbers: [`PAPER_NUMBERS.json`](../PAPER_NUMBERS.json)  
> Do NOT use individual experiment files directly in the paper — always refer to `PAPER_NUMBERS.json`.

---

## 📋 Quick Reference — Paper Numbers

| Metric | Value | Source File |
|--------|-------|-------------|
| Mean CPA (GPT-2-XL) | **0.0666** | `cdd_summary.json` |
| Mean Probe Accuracy | **91.45%** | `cdd_summary.json` |
| Mean CRR | **23.1×** (100% > 0.70) | `cdd_summary.json` |
| Circuit Specificity | **35.9×** | `cdd_summary.json` |
| Circuit Ablation Drop | **546%** | `cdd_summary.json` |
| Regime MW p-value | **0.0103** | `cdd_summary.json` |
| Context Paradox r | **−0.781** (p=0.038) | `cdd_summary.json` |
| Pythia-1.4B CPA | **0.1005** | `replication_pythia.json` |
| Pythia-410M CPA | **0.0473** | `replication_pythia.json` |
| CREAK CPA | **0.0148** | `creak_validation.json` |
| Ablation A CPA range | **0.0160** (STABLE) | `ablation_A_K_sensitivity_fixed.json` |
| Ablation C CPA range | **0.0054** (STABLE) | `ablation_C_probe_reg.json` |

---

## 📁 File Directory

### ✅ VERIFIED CORE RESULTS — Use These

| File | What It Contains | Used In Paper |
|------|-----------------|---------------|
| `cdd_summary.json` | Master summary: CPA, CRR, circuit, domain regimes | §4, §5 all sections |
| `replication_pythia.json` | Pythia-1.4B + Pythia-410M CPA replication | §5.5 Replication |
| `creak_validation.json` | Out-of-domain CREAK dataset validation | §5.5 Replication |
| `bootstrap_stability.json` | Per-domain regime stability (N=2000 samples) | Appendix A |
| `autotruth_domain_regimes.csv` | 22 domains with IE, CPA, regime labels | §5.4 + Table 4 |
| `autotruth_circuit.csv` | Top-50 circuit components with scores | §5.1 + Table |
| `cpa_results.json` | Per-component CPA values (50 components) | §5.3 |
| `crr_results.json` | Per-component CRR values | §5.2 |
| `ablation_A_K_sensitivity_fixed.json` | K sensitivity: CPA stable across K=10–100 | §5.6 Ablation A |
| `ablation_B_stage2_bypass.json` | Stage 2 necessity: cPCA vs mean-diff vs random | §5.6 Ablation B |
| `ablation_C_probe_reg.json` | Probe regularisation: CPA stable across C | §5.6 Ablation C |
| `autotruth_final_results.json` | Final aggregated results | §5 summary |
| `final_results.json` | Same as above (later version) | §5 summary |

---

### 📊 RAW EXPERIMENT DATA — For Reproducibility

| File | What It Contains |
|------|------------------|
| `all_scores.csv` | All 1248 components with IE, consistency, score (GPT-2-XL, n=200) |
| `autotruth_all_scores.csv` | Earlier version of all_scores (n_samples different) |
| `autotruth_all_scores_v2.csv` | V2 iteration during development |
| `autotruth_circuit_v2.csv` | Circuit components from v2 run |
| `cdd_components_gpt2-xl.csv` | Top-50 circuit components with CPA+CRR (GPT-2-XL) |
| `cdd_components_gpt2_xl.csv` | Duplicate of above (different naming) |
| `cdd_components_eleutherai_pythia_410m_deduped.csv` | Top-30 circuit components (Pythia-410M) |
| `cdd_summary_gpt2-xl.csv` | CSV version of cdd_summary |
| `autotruth_domain_CPA.csv` | Per-domain CPA values |
| `autotruth_cross_domain.csv` | Cross-domain analysis raw |
| `cross_domain.csv` | Same (earlier naming) |
| `cross_model_comparison.csv` | Side-by-side model comparison |
| `circuit.csv` | Full circuit data (all versions) |
| `autotruth_antisuppression_circuit.csv` | Anti-suppression sub-circuit |
| `replication_gpt2_xl.json` | GPT-2-XL replication entry |
| `replication_eleutherai_pythia_410m_deduped.json` | Pythia-410M standalone replication |

---

### 🖼️ FIGURES

| File | What It Is | Status |
|------|-----------|--------|
| `CDD_main_figure_final.png` | Main 4-panel CDD figure | ✅ Use in paper |
| `CDD_three_regime_final.png` | Three-regime scatter plot | ✅ Use in paper |
| `CDD_two_regime_figure.png` | Earlier 2-regime version | ❌ Superseded |
| `autotruth_circuit_heatmap_final.png` | Circuit heatmap (48 layers × 25 heads) | ✅ Use in paper |
| `autotruth_circuit_heatmap.png` | Earlier circuit heatmap | ❌ Superseded |
| `autotruth_heatmap_v2.png` | V2 heatmap | ❌ Superseded |
| `autotruth_main_figure.png` | Earlier main figure | ❌ Superseded |

---

### ⚙️ CODE FILES (should be in `src/`, not here)

| File | What It Does |
|------|-------------|
| `experiment.ipynb` | Main experiment notebook (1.4MB) |
| `activation_patching.py` | Stage 1: activation patching |
| `cpca_crr_cpa.py` | Stage 2+3: cPCA, CRR, CPA computation |
| `contrastive_pairs.py` | Dataset pair construction |
| `model_loader.py` | HookedTransformer loading utility |

---

### 🗑️ SUPERSEDED / DUPLICATE FILES (safe to ignore)

| File | Reason |
|------|--------|
| `ablation_A_K_sensitivity.json` | Broken pipeline values — use `_fixed.json` instead |
| `ablation_A_K_sensitivity.csv` | Same — broken |
| `ablation_C_probe_regularization.json` | Duplicate of `ablation_C_probe_reg.json` |
| `ablation_C_probe_regularization.csv` | Duplicate |
| `ablation_C_probe_reg.csv` | Duplicate |
| `ablation_stats.json` | Intermediate stats from broken run |
| `autotruth_results_summary.json` | Incomplete early summary |
| `autotruth_results_v2.json` | Superseded by `final_results.json` |

---

## 🗂️ Recommended Folder Structure (Future)

For reproducibility, these files should eventually live in:
```
AutoTruth/
├── PAPER_NUMBERS.json          ← single source of truth
├── results/
│   ├── core/                   ← verified final results
│   ├── ablations/              ← ablation study results
│   ├── replication/            ← cross-model/dataset results
│   ├── raw/                    ← raw experiment data
│   └── figures/                ← final paper figures
├── src/                        ← all .py and .ipynb code
└── paper/                      ← LaTeX
```
