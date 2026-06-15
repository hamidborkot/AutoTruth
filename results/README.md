# Results Directory

All pre-computed results from the AutoTruth paper experiments.

## Files

| File | Description | Rows | Columns |
|---|---|---|---|
| `autotruth_all_scores_v2.csv` | All 1248 component scores | 1248 | component, type, layer, head, mean_ie, consistency, score |
| `autotruth_circuit_v2.csv` | Top-50 promoting circuit | 50 | Same as above |
| `autotruth_antisuppression_circuit.csv` | Suppressing subnetwork | ~20 | Same as above |
| `autotruth_cross_domain.csv` | Per-category breakdown | 22 | category, n, mean_ie, std_ie, pct_positive |
| `autotruth_final_results.json` | Complete summary | — | All metrics |

## Experiment Details

- **Model**: GPT-2-XL (48 layers, 25 heads, d_model=1600)
- **Dataset**: TruthfulQA generation split (817 samples, 200 used)
- **Hardware**: NVIDIA RTX 5070 Laptop GPU (8.55 GB VRAM)
- **Runtime**: ~5 hours 11 minutes
- **Seed**: 42
- **Date**: June 2026

## Key Finding

The truthfulness circuit in GPT-2-XL is dominated by **MLP layers** (100% of top-50 circuit),
with **L00_MLP** as the dominant component (score 0.8118, 22x stronger than the second component).
This circuit is **causally validated** at 35.9x specificity (p<0.0001).
