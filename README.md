# AutoTruth + CDD: Causal Direction Decomposition for Truthfulness in LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: GPT-2-XL](https://img.shields.io/badge/Model-GPT--2--XL-orange.svg)](https://huggingface.co/gpt2-xl)
[![Dataset: TruthfulQA](https://img.shields.io/badge/Dataset-TruthfulQA-red.svg)](https://huggingface.co/datasets/truthful_qa)
[![Status: AAAI-27 Submission](https://img.shields.io/badge/Status-AAAI--27%20Submission-purple.svg)](#)
[![Paper](https://img.shields.io/badge/Paper-CDD%20Draft-blue.svg)](paper/CDD_Full_Paper_Draft.md)

> **AutoTruth** discovers and validates the causal truthfulness circuit in GPT-2-XL via contrastive activation patching. **CDD (Causal Direction Decomposition)** then decomposes each circuit component into its specific causal directions, proves that linear probes measure a fundamentally different structure (CPA = 0.0666), and reveals a three-regime theory of domain-selective hallucination — all submitted to **AAAI 2027**.

---

## TL;DR

Two dominant methods study LLM truthfulness:
- **Probing** finds directions that *correlate* with truth — but never proves causation
- **Circuit discovery** proves causality at the component level — but cannot say *which direction within a component* is doing the work

**This repo bridges both.** We first discover the truthfulness circuit (AutoTruth), then decompose each component into causal directions (CDD), compare them to probe directions, and explain why the same model is truthful on Advertising but hallucinates on Conspiracies.

---

## Key Results

### AutoTruth — Truthfulness Circuit (Stage 1)

| Metric | Value |
|---|---|
| Circuit size | 50 / 1,248 components (4.0% of GPT-2-XL) |
| Specificity ratio | **35.9×** (circuit vs. random ablation) |
| Statistical significance | **p < 0.0001** (paired t-test, n = 200) |
| Circuit ablation drop | 546% logit difference reduction |
| Random ablation drop | 15.2% (confirms specificity) |
| Dominant component | L00_MLP (IE = 0.811, 22× stronger than #2) |
| Architecture | 100% MLP-dominated |

### CDD — Causal Direction Decomposition (Stages 2–4)

| Metric | Value | Meaning |
|---|---|---|
| Mean CPA | **0.0666** | Probes share only 6.7% directional alignment with causal directions |
| Mean probe accuracy | **91.45%** | Probe is highly accurate yet measures different structure |
| Mean CRR | **23.1** | Rank-1 patching recovers 2310% of full-component IE |
| Components CRR ≥ 0.70 | **100%** (50/50) | Causal effect is hyper-concentrated in top direction |
| Mean effective rank | **57.74** | Causal variation spans ~58 directions per component |
| Regime separation | **p = 0.0103** | Three regimes are statistically distinct |
| Context-Driven paradox | **r = −0.781, p = 0.038** | Global direction alignment *hurts* in context-sensitive domains |

### Three-Regime Theory (Novel Discovery)

| Regime | Domains | Mean IE | Mean CPA | Explanation |
|---|---|---|---|---|
| **Direction-Driven** | Advertising, Fiction, Distraction | 0.617 | 0.703 | Global causal direction active; universal facts |
| **Context-Driven** | Indexical Time, Subjective, Identity, Myths, Religion, Stereotypes, Location | 0.686 | 0.324 | Domain-local directions; global direction *hurts* (r=−0.781) |
| **Circuit Failure** | Conspiracies, Logical Falsehood, Paranormal, Misconceptions, Nutrition, Misquotations, Superstitions, Education | 0.149 | 0.220 | No stable causal direction in any regime |

---

## Repository Structure

```
AutoTruth/
├── README.md                            # This file
├── requirements.txt                     # All dependencies
├── LICENSE                              # MIT License
├── REPRODUCIBILITY.md                   # Full step-by-step reproduction guide
├── CONTRIBUTING.md                      # Contribution guidelines
│
├── paper/
│   └── CDD_Full_Paper_Draft.md          # Complete AAAI 2027 paper draft
│
├── notebooks/
│   ├── 01_environment_setup.ipynb       # Install & configure environment
│   ├── 02_data_pipeline.ipynb           # Model load, data, tokenizer
│   ├── 03_mean_activations.ipynb        # Baseline & mean activation cache
│   ├── 04_patching_loop.ipynb           # Core AutoTruth patching experiment
│   ├── 05_scoring_circuit.ipynb         # Score, rank & save circuit
│   ├── 06_ablation_validation.ipynb     # Mean ablation + statistics
│   ├── 07_anti_circuit.ipynb            # Suppressing subnetwork
│   ├── 08_cross_domain.ipynb            # Per-category analysis (22 domains)
│   ├── 09_visualizations.ipynb          # AutoTruth figures
│   ├── 10_cdd_cpca.ipynb                # CDD Stage 2: Contrastive PCA
│   ├── 11_cdd_rank1_crr.ipynb           # CDD Stage 3a: Rank-1 CRR
│   ├── 12_cdd_cpa.ipynb                 # CDD Stage 3b: Probe-Circuit CPA
│   ├── 13_cdd_domain_conditional.ipynb  # CDD Stage 4: Domain-conditional analysis
│   └── 14_cdd_three_regimes.ipynb       # CDD Stage 5: Three-regime discovery
│
├── src/
│   └── autotruth/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── patching.py
│       ├── scoring.py
│       ├── ablation.py
│       ├── cdd/
│       │   ├── __init__.py
│       │   ├── cpca.py                  # Contrastive PCA decomposition
│       │   ├── rank1.py                 # Rank-1 causal intervention + CRR
│       │   ├── probe.py                 # Probe training + CPA computation
│       │   └── domain.py               # Domain-conditional CDD + regime clustering
│       └── visualization.py
│
├── results/
│   ├── autotruth_all_scores_v2.csv
│   ├── autotruth_circuit_v2.csv
│   ├── autotruth_antisuppression_circuit.csv
│   ├── autotruth_cross_domain.csv
│   ├── autotruth_final_results.json
│   ├── autotruth_domain_CPA.csv         # Domain-conditional CPA scores
│   └── autotruth_domain_regimes.csv     # Domain regime classification
│
└── figures/
    ├── autotruth_main_figure.png
    ├── autotruth_circuit_heatmap_final.png
    ├── CDD_main_figure_final.png        # CDD 4-panel result
    ├── CDD_two_regime_figure.png
    └── CDD_three_regime_final.png       # Final three-regime figure
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/hamidborkot/AutoTruth.git
cd AutoTruth
pip install -r requirements.txt
```

### 2. Run AutoTruth (Circuit Discovery)

```bash
python src/run_experiment.py --model gpt2-xl --samples 200 --top_k 50
```

Expected output:
```
Specificity ratio : 35.9×
p-value           : <0.0001
Circuit size      : 50/1248 (4.0%)
Circuit drop      : 546%
Random drop       : 15.2%
```

### 3. Run CDD (Causal Direction Decomposition)

```bash
python src/run_cdd.py --circuit results/autotruth_circuit_v2.csv --pairs results/pairs.pkl
```

Expected CDD output:
```
Mean effective rank  : 57.74
Mean CRR             : 23.101  (100% components above 0.70)
Mean CPA             : 0.0666
Mean probe accuracy  : 0.9145
Regime separation p  : 0.0103
Context-Driven r     : -0.781 (p=0.038)
```

### 4. Hardware Requirements

| Resource | Minimum | Used in Paper |
|---|---|---|
| GPU VRAM | 8 GB | RTX 5070 Laptop (8.55 GB) |
| RAM | 16 GB | 16 GB |
| Storage | 5 GB | ~4 GB |
| Time — AutoTruth | ~5 hours | 5h 11min |
| Time — CDD (all stages) | ~8 hours | ~8h on RTX 5070 |

---

## Reproducing All Results

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact step-by-step instructions
with expected outputs at every stage.

---

## Paper

Full AAAI 2027 paper draft: [`paper/CDD_Full_Paper_Draft.md`](paper/CDD_Full_Paper_Draft.md)

**Title:** Causal Direction Decomposition (CDD): Three Mechanistic Regimes of Truthfulness in Large Language Models

**Core findings:**
1. Linear probes (91.45% accuracy) share only 6.66% directional alignment with causally active circuit directions — probes measure correlational structure, not causal
2. Rank-1 causal patching recovers 2310% of full-component IE (mean CRR = 23.1) — truthfulness causality is hyper-concentrated in single directions
3. Domain-conditional CDD reveals three mechanistic regimes; within Context-Driven domains, global direction alignment *negatively* predicts performance (r = −0.781, p = 0.038)

---

## Citation

```bibtex
@inproceedings{tulla2027cdd,
  title     = {Causal Direction Decomposition ({CDD}): Three Mechanistic Regimes
               of Truthfulness in Large Language Models},
  author    = {Tulla, Md. Hamid Borkot},
  booktitle = {Proceedings of the Forty-First {AAAI} Conference on Artificial Intelligence},
  year      = {2027},
  address   = {Chongqing, China},
  note      = {AI Alignment Track}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contact

**Md. Hamid Borkot Tulla** — GitHub: [@hamidborkot](https://github.com/hamidborkot)
