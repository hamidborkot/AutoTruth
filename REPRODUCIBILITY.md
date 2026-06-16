# Reproducibility Guide — AutoTruth + CDD

This guide allows any researcher to reproduce every result in the paper
**"Causal Direction Decomposition (CDD): Three Mechanistic Regimes of Truthfulness in LLMs"**
from scratch on a single consumer GPU.

---

## Environment

| Item | Specification |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop, 8.55 GB VRAM |
| RAM | 16 GB |
| Python | 3.10+ |
| CUDA | 12.1 |
| PyTorch | 2.x |
| TransformerLens | 1.x |

```bash
git clone https://github.com/hamidborkot/AutoTruth.git
cd AutoTruth
pip install -r requirements.txt
```

---

## Stage 1 — AutoTruth: Circuit Discovery

**Notebook:** `04_patching_loop.ipynb` + `05_scoring_circuit.ipynb`

```bash
python src/run_experiment.py --model gpt2-xl --samples 200 --seed 42 --top_k 50
```

**Expected outputs:**
```
Specificity ratio : 35.9×
p-value           : < 0.0001
Circuit size      : 50 / 1248 (4.0%)
Circuit drop      : 546%
Random drop       : 15.2%
Dominant component: L00_MLP (IE = 0.811)
```

Output files: `results/autotruth_circuit_v2.csv`, `results/autotruth_all_scores_v2.csv`

---

## Stage 2 — Cross-Domain Analysis

**Notebook:** `08_cross_domain.ipynb`

Key expected values: Advertising IE=+1.247, Indexical Time IE=+1.237, Conspiracies IE=−0.019, Logical Falsehood IE=−0.048

Output: `results/autotruth_cross_domain.csv`

---

## Stage 3 — CDD: Contrastive PCA

**Notebook:** `10_cdd_cpca.ipynb`

**Expected outputs:**
```
Mean effective rank  : 57.74
Median effective rank: 77.5
Max effective rank   : 88
Components with rank=1: 0/50
```

Note: High effective rank (57.74) is the correct result, not an error.
It shows causal variation is distributed across many directions — motivating rank-1 validation.

---

## Stage 4 — CDD: Rank-1 CRR

**Notebook:** `11_cdd_rank1_crr.ipynb`

**Expected outputs:**
```
Mean CRR : 23.101
Components with CRR >= 0.70 : 100.0% (50/50)
Top component: L33H05 CRR=45.695
```

Note: CRR > 1.0 is correct — rank-1 patching overshoots full-component IE because
mean ablation dilutes the causal signal; rank-1 isolation removes this dilution.

---

## Stage 5 — CDD: Probe-Circuit CPA (Central Result)

**Notebook:** `12_cdd_cpa.ipynb`

**Expected outputs:**
```
Mean CPA (causal-probe alignment) : 0.0666
Mean probe accuracy               : 0.9145
Result: STRONG MISALIGNMENT
```

**This is H1 confirmed.** CPA = 0.0666 is the paper's central finding — not an error.
A probe achieving 91.45% accuracy shares only 6.66% directional alignment with the
causally active direction within the same component.

---

## Stage 6 — CDD: Domain-Conditional Analysis

**Notebook:** `13_cdd_domain_conditional.ipynb`

**Expected outputs (selected):**
```
Advertising           IE=+1.247  CPA=0.660
Fiction               IE=+0.358  CPA=0.811
Indexical Time        IE=+1.237  CPA=0.202
Conspiracies          IE=-0.019  CPA=0.216
Logical Falsehood     IE=-0.048  CPA=0.189
Global Pearson r = 0.2342, p = 0.3496  (not significant — two regimes mixed)
```

Note: The global correlation is non-significant intentionally — two regimes with opposite
relationships are mixed. The structure only emerges in Stage 7.

---

## Stage 7 — CDD: Three-Regime Discovery

**Notebook:** `14_cdd_three_regimes.ipynb`

**Expected outputs:**
```
Direction-Driven (n=3): Advertising, Fiction, Distraction
  Mean IE=0.617, Mean CPA=0.703

Context-Driven (n=7): Indexical Time, Subjective, Identity, Myths, Religion, Stereotypes, Location
  Mean IE=0.686, Mean CPA=0.324
  Within-regime r = -0.781, p = 0.038  ← Context-Driven Paradox

Circuit Failure (n=8): Conspiracies, Logical Falsehood, Paranormal, Misconceptions,
                       Nutrition, Misquotations, Superstitions, Education
  Mean IE=0.149, Mean CPA=0.220

Mann-Whitney U p = 0.0103  ← Regimes are statistically distinct
```

---

## Complete Results Summary

| Stage | Key Metric | Value |
|---|---|---|
| AutoTruth | Specificity ratio | 35.9× |
| AutoTruth | p-value | < 0.0001 |
| AutoTruth | Circuit size | 50/1,248 (4.0%) |
| cPCA | Mean effective rank | 57.74 |
| Rank-1 CRR | Mean CRR | 23.101 |
| Rank-1 CRR | % above 0.70 | 100% |
| **CPA (H1)** | **Mean CPA** | **0.0666** |
| CPA | Probe accuracy | 91.45% |
| Regimes | Mann-Whitney p | 0.0103 |
| Context-Driven | Pearson r | −0.781 (p=0.038) |

---

## Common Issues

**CPA near zero:** This is the expected and confirmed result (0.0666).

**CRR > 1.0:** This is correct — see Stage 4 note above.

**Global domain correlation not significant:** Expected — see Stage 6 note above.

**Domain categories skipped:** Health (n=2), Proverbs (n=2), Misconceptions: Topical (n=2)
are excluded due to insufficient pairs. This is a documented limitation in the paper.

**CUDA OOM:** Ensure no other GPU processes are running. Patching runs one component at a time by default.
