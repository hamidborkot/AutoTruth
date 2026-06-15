# Reproducibility Guide — AutoTruth

This document provides exact instructions to reproduce every result in the paper from scratch.

---

## Environment

| Component | Version Used in Paper |
|---|---|
| OS | Windows 11 / Linux |
| Python | 3.10+ |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.41.0 |
| TransformerLens | 1.19+ |
| CUDA | 12.8 |
| GPU | NVIDIA RTX 5070 Laptop (8.55 GB VRAM) |
| RAM | 16 GB |
| Random Seed | 42 |

---

## Step 1: Setup

```bash
git clone https://github.com/hamidborkot/AutoTruth.git
cd AutoTruth
python -m venv autotruth_env
source autotruth_env/bin/activate  # Windows: autotruth_env\Scripts\activate
pip install -r requirements.txt
```

---

## Step 2: Dataset

TruthfulQA is loaded automatically from HuggingFace — no manual download needed:
```python
from datasets import load_dataset
dataset = load_dataset("truthful_qa", "generation", split="validation")
# 817 samples, columns: type, category, question, best_answer,
# correct_answers, incorrect_answers, source
```

---

## Step 3: Model

GPT-2-XL is loaded automatically via TransformerLens — no HF token required:
```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-xl", device="cuda")
# 48 layers, 25 heads/layer, d_model=1600
# VRAM usage: ~6.74 GB
```

---

## Step 4: Run Experiments

### Option A — Full Pipeline (Automated)
```bash
python src/run_experiment.py --model gpt2-xl --samples 200 --top_k 50 --seed 42
```
Estimated time: ~5 hours on RTX 5070 (8GB VRAM)

### Option B — Step by Step (Notebooks)
Run notebooks in this exact order:

| Notebook | Purpose | Est. Time |
|---|---|---|
| `01_environment_setup.ipynb` | Install + verify CUDA | 2 min |
| `02_data_pipeline.ipynb` | Load model + TruthfulQA | 5 min |
| `03_mean_activations.ipynb` | Compute dataset mean activations | 15 min |
| `04_patching_loop.ipynb` | **Core experiment** (activation patching) | ~5 hours |
| `05_scoring_circuit.ipynb` | Score all 1248 components | 1 min |
| `06_ablation_validation.ipynb` | Mean ablation + t-test | 20 min |
| `07_anti_circuit.ipynb` | Suppressing subnetwork analysis | 5 min |
| `08_cross_domain.ipynb` | Per-category breakdown | 15 min |
| `09_visualizations.ipynb` | Generate all paper figures | 2 min |

---

## Step 5: Expected Outputs

### results/ directory
```
autotruth_all_scores_v2.csv          # 1248 rows, columns: component, type, layer, head, mean_ie, consistency, score
autotruth_circuit_v2.csv             # 50 rows (top circuit components)
autotruth_antisuppression_circuit.csv # Suppressing subnetwork
autotruth_cross_domain.csv           # 22 rows (one per TruthfulQA category)
autotruth_final_results.json         # Complete results summary
```

### figures/ directory
```
autotruth_main_figure.png            # 4-panel main figure (300 DPI)
autotruth_circuit_heatmap_final.png  # Layer x Head heatmap (300 DPI)
```

---

## Step 6: Verify Results

Your reproduced results should match within ±2% due to floating-point variance:

| Metric | Paper Value | Acceptable Range |
|---|---|---|
| Specificity ratio | 35.9x | 30x – 40x |
| p-value | <0.0001 | <0.001 |
| Circuit size | 50 | 45–55 |
| Circuit drop % | 546% | 400%–650% |
| Random drop % | 15.2% | 10%–25% |
| Top component | L00_MLP | L00_MLP |

---

## Contrastive Pair Construction

For each TruthfulQA item:
- **Truthful prompt**: `"Q: {question}\nA: {correct_answers[0]}"`
- **Hallucinated prompt**: `"Q: {question}\nA: {incorrect_answers[0]}"`
- Items with empty correct or incorrect answer lists are skipped
- Maximum sequence length: 128 tokens (truncated)

---

## Circuit Scoring Formula

```
AutoTruth Score = mean_IE × Consistency

where:
  mean_IE     = mean Indirect Effect across all pairs
  Consistency = fraction of pairs where IE > 0
  Circuit     = components with score > mean(all scores) + 1×std(all scores)
```

---

## Ablation Protocol

- **Mean ablation** (not zero ablation): replaces component activation with dataset mean
- **Circuit ablation**: ablate all 50 circuit components simultaneously
- **Random ablation**: average of 5 random seeds, each ablating 50 random components
- **Validation subset**: 50 pairs (stratified from full 200)
- **Statistical test**: paired two-sided t-test (baseline LD vs circuit ablation LD)

---

## Known Issues & Notes

1. **Zero ablation gives incorrect results** (5327% drop) — always use mean ablation
2. **Patching direction matters** — patch corrupt→clean activation in clean run (not clean→corrupt)
3. **Session interruptions**: the patching loop (Cell 9) takes ~5 hours; use VS Code autosave or checkpoint every 50 pairs
4. **VRAM**: with 8.55 GB, VRAM usage peaks at ~8.1 GB during patching — close other GPU processes

---

## Contact

For reproducibility issues, open a GitHub Issue at:
https://github.com/hamidborkot/AutoTruth/issues
