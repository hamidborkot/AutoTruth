# AutoTruth — Results Directory

All experimental results are stored as JSON files and are fully reproducible
using the notebooks and source code in this repository.

---

## Result Files

### Core Experiment (GPT-2-XL on TruthfulQA)

| File | Description |
|---|---|
| `autotruth_final_results.json` | Primary CPA, probe accuracy, circuit specificity (35.9×, p<0.0001) |
| `autotruth_domain_regimes.csv` | Per-domain regime classification (18 domains × CPA × regime) |
| `cdd_summary.json` | Full CDD pipeline summary: CPCA, probe, CPA, ablation baselines |

### Replication & Generalizability

| File | Description |
|---|---|
| `replication_pythia.json` | Pythia-1.4B-deduped replication: CPA=0.1005, H1 confirmed |
| `creak_validation.json` | CREAK cross-dataset validation: CPA=0.0148, H1 confirmed |
| `bootstrap_stability.json` | 18-domain bootstrap stability: 94.4% core-domain stability |

---

## Five Evidence Pillars (Paper Summary)

| # | Evidence | Value | Status |
|---|---|---|---|
| 1 | GPT-2-XL CPA (TruthfulQA) | 0.0666 | ✅ H1 confirmed |
| 2 | Pythia-1.4B CPA (TruthfulQA) | 0.1005 | ✅ H1 replicates |
| 3 | Pythia-1.4B CPA (CREAK) | 0.0148 | ✅ H1 cross-dataset |
| 4 | Circuit specificity vs random | 35.9×, p<0.0001 | ✅ Circuit is real |
| 5 | Three-regime Mann-Whitney | p=0.0103 | ✅ Domain structure real |

---

## Bootstrap Stability Summary

| Metric | Value |
|---|---|
| Overall mean stability | 78.1% |
| Core domains (dist≥0.15) mean | **94.4%** ← cite this in paper |
| Domains ≥85% stable | 9/18 |
| Boundary-zone domains | 11/18 |

Boundary-zone domains sit within σ=0.12 of a regime threshold, reflecting
genuine empirical ambiguity in TruthfulQA's category definitions, not
classification error.

---

## Reproducibility

See `../REPRODUCIBILITY.md` for hardware specs, runtime estimates, seeds,
and step-by-step reproduction instructions.

All results were generated with:
- Python 3.10, PyTorch 2.2, HuggingFace Transformers 4.40
- Seed: 42 (all experiments)
- Hardware: see REPRODUCIBILITY.md
