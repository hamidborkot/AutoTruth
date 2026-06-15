# AutoTruth: Automated Circuit Discovery for Truthfulness Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: GPT-2-XL](https://img.shields.io/badge/Model-GPT--2--XL-orange.svg)](https://huggingface.co/gpt2-xl)
[![Dataset: TruthfulQA](https://img.shields.io/badge/Dataset-TruthfulQA-red.svg)](https://huggingface.co/datasets/truthful_qa)
[![Status: AAAI-27 Submission](https://img.shields.io/badge/Status-AAAI--27%20Submission-purple.svg)](#)

> **AutoTruth** is the first automated pipeline for discovering and validating causal circuits specifically responsible for truthfulness in large language models, using contrastive activation patching on semantically controlled truthful/hallucinated input pairs.

---

## Key Results (GPT-2-XL, 200 samples, TruthfulQA)

| Metric | Value |
|---|---|
| **Specificity Ratio** | **35.9x** (circuit vs. random ablation) |
| **Statistical Significance** | p < 0.0001 (paired t-test) |
| **Circuit Size** | 50 / 1248 components (4.0% of model) |
| **Circuit Drop** | 546% logit difference reduction |
| **Random Drop** | 15.2% (confirms specificity) |
| **Dominant Component** | L00_MLP (score = 0.8118, 22x stronger than #2) |
| **Cross-Domain Coverage** | 22 TruthfulQA categories |
| **Perfect Categories** | Advertising, Indexical Time, Education (100%) |
| **Failing Categories** | Health (0%), Conspiracies (negative IE) |

---

## What Is AutoTruth?

Large language models (LLMs) hallucinate — they generate false information confidently. We know **that** they lie, but not **which internal components** cause the lie.

AutoTruth solves this by:
1. Building **semantically controlled contrastive pairs** (truthful vs. hallucinated inputs)
2. Running **activation patching** to measure each component's causal contribution to truthfulness
3. **Automatically ranking** all 1,248 components by an IE × Consistency score
4. **Validating causality** via mean ablation (not zero-ablation) with statistical testing

### The Core Novelty
- **First automated, truthfulness-specific circuit discovery pipeline**
- **Dual-circuit finding**: promoting circuit (4% of model) + suppressing subnetwork
- **Cross-domain validation** across 22 categories
- **Statistically rigorous**: p < 0.0001, 35.9x specificity ratio
- **MLP dominance finding**: top-10 circuit components are all MLP layers, extending Meng et al. (ROME, 2022) from knowledge storage to truthfulness expression

---

## Repository Structure

```
AutoTruth/
├── README.md                          # This file
├── requirements.txt                   # All dependencies
├── LICENSE                            # MIT License
├── REPRODUCIBILITY.md                 # Step-by-step reproduction guide
├── CONTRIBUTING.md                    # Contribution guidelines
│
├── notebooks/
│   ├── 01_environment_setup.ipynb     # Cell 1-2: Install & Config
│   ├── 02_data_pipeline.ipynb         # Cell 3-6: Model, Data, Tokenizer
│   ├── 03_mean_activations.ipynb      # Cell 7-8: Baseline & Mean Cache
│   ├── 04_patching_loop.ipynb         # Cell 9: Core AutoTruth Experiment
│   ├── 05_scoring_circuit.ipynb       # Cell 10: Score, Rank & Save
│   ├── 06_ablation_validation.ipynb   # Cell 11: Mean Ablation + Stats
│   ├── 07_anti_circuit.ipynb          # Cell 12: Suppressing Subnetwork
│   ├── 08_cross_domain.ipynb          # Cell 13: Per-Category Analysis
│   └── 09_visualizations.ipynb        # Final Cells A-C: All Figures
│
├── src/
│   ├── autotruth/
│   │   ├── __init__.py
│   │   ├── config.py                  # Centralized config
│   │   ├── data.py                    # Contrastive pair builder
│   │   ├── patching.py                # Activation patching engine
│   │   ├── scoring.py                 # Circuit scoring & ranking
│   │   ├── ablation.py                # Mean ablation validator
│   │   └── visualization.py           # All plotting functions
│   └── run_experiment.py              # Single-command full pipeline
│
├── results/
│   ├── autotruth_all_scores_v2.csv    # All 1248 component scores
│   ├── autotruth_circuit_v2.csv       # Top-50 promoting circuit
│   ├── autotruth_antisuppression_circuit.csv  # Suppressing subnetwork
│   ├── autotruth_cross_domain.csv     # Per-category IE breakdown
│   └── autotruth_final_results.json   # Complete results summary
│
└── figures/
    ├── autotruth_main_figure.png      # 4-panel main result figure
    └── autotruth_circuit_heatmap_final.png  # Layer x head heatmap
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/hamidborkot/AutoTruth.git
cd AutoTruth
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Single Command)
```bash
python src/run_experiment.py --model gpt2-xl --samples 200 --top_k 50
```

### 3. Run Step by Step (Notebooks)
Open notebooks in order `01_` → `09_` in Jupyter or VS Code.

### 4. Hardware Requirements
| Resource | Minimum | Used in Paper |
|---|---|---|
| GPU VRAM | 8 GB | RTX 5070 Laptop (8.55 GB) |
| RAM | 16 GB | 16 GB |
| Storage | 5 GB | ~3 GB |
| Time (200 samples) | ~5 hours | 5h 11min on RTX 5070 |

---

## Reproducing Paper Results

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact step-by-step instructions.

All results are pre-computed in `results/`. To verify from scratch:
```bash
python src/run_experiment.py --model gpt2-xl --samples 200 --seed 42
```
Expected output:
```
Specificity ratio : 35.9x
p-value           : <0.0001
Circuit size      : 50/1248 (4.0%)
Circuit drop      : 546%
Random drop       : 15.2%
```

---

## Citation

If you use AutoTruth in your research, please cite:
```bibtex
@inproceedings{tulla2027autotruth,
  title     = {AutoTruth: Automated Circuit Discovery for Truthfulness Verification
               in Small Language Models via Contrastive Activation Patching},
  author    = {Tulla, Md. Hamid Borkot},
  booktitle = {Proceedings of the Forty-First AAAI Conference on Artificial Intelligence},
  year      = {2027},
  address   = {Montr{\'e}al, Canada}
}
```

---

## License
MIT License. See [LICENSE](LICENSE).

---

## Contact
**Md. Hamid Borkot Tulla** — GitHub: [@hamidborkot](https://github.com/hamidborkot)
