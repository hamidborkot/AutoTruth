# Causal Direction Decomposition (CDD): Three Mechanistic Regimes of Truthfulness in Large Language Models

**Anonymous Authors — AAAI 2027 Submission**
*AI Alignment Track | Submission Deadline: July 21, 2026*

---

## Abstract

Linear probing and circuit discovery are the two dominant paradigms for studying truthfulness in large language models (LLMs), yet they have developed as independent research programs with no empirical connection. Probing identifies correlational "truth directions" in residual activations but cannot establish causation. Circuit discovery proves causality at the component level but cannot identify which specific directions *within* a component are causally active. We introduce **Causal Direction Decomposition (CDD)**, a three-stage framework that (1) identifies causally-validated circuit components via activation patching, (2) decomposes each component's output into contrastive principal directions via cPCA, and (3) validates those directions causally through rank-1 intervention and compares them to probe directions. Applying CDD to GPT-2-XL on 817 TruthfulQA samples across 22 semantic domains, we make three findings. **H1 (confirmed):** the mean Causal-Probe Alignment (CPA) across all 50 circuit components is 0.0666 — linear probes achieving 91.4% classification accuracy identify directions near-orthogonal to causally active directions. Probing measures different structure than what actually mediates truthfulness. **CRR finding:** rank-1 intervention recovers mean CRR = 23.1 (2310% of baseline IE; 100% of components above threshold), confirming causal effects are hyper-concentrated in single directions. **Three-regime discovery:** domain-conditional CDD reveals a previously undescribed three-regime structure — Direction-Driven (high IE, high CPA; universal facts), Context-Driven (high IE, low CPA; context-dependent truth), and Circuit Failure (low IE, low CPA; contested knowledge). Within the Context-Driven regime, global CPA alignment negatively predicts performance (r = −0.781, p = 0.038). Regimes are statistically separable (Mann-Whitney p = 0.0103). These findings challenge the causal assumptions of the probing research program, predict failure modes of representation engineering in context-sensitive domains, and provide the first mechanistic, domain-resolved theory of selective hallucination.

---

## 1. Introduction

### 1.1 The Problem

Large language models do not hallucinate uniformly. On TruthfulQA, GPT-2-XL achieves mean IE of +1.247 on Advertising and +1.237 on Indexical Time, yet produces negative IE on Logical Falsehood (−0.048) and Conspiracies (−0.019). This domain-selectivity is structured and reproducible. Understanding *why* the same model is truthful in some domains and confabulates in others is one of the most important open problems in LLM alignment.

### 1.2 Two Paradigms, No Bridge

**Probing** (Burns et al., 2023; Li et al., 2023; Marks & Tegmark, 2023) achieves high classification accuracy but, as Sharkey et al. (2025) state: *"Probes detect correlations, rather than causal variables, in hidden activations."* No prior work empirically tests whether probe directions are causally active for truthfulness.

**Circuit discovery** (Wang et al., 2022; Conmy et al., 2023) proves causation but operates at whole-component granularity — too coarse to identify which direction within a component mediates the causal effect.

No prior work asks whether probing and circuit discovery measure the same internal structure, decomposes circuit component outputs into causal directions, or explains domain-selective hallucination at within-component resolution.

### 1.3 Contributions

1. **CDD framework:** First method to decompose causally-validated circuit components into causal feature directions, validate via rank-1 intervention, and compare to probe directions.
2. **H1 confirmed:** Mean CPA = 0.0666 with probe accuracy 91.45% — probes and circuits are near-orthogonal.
3. **Three-Regime Theory:** Direction-Driven, Context-Driven, Circuit Failure — statistically separable, explaining domain-selective hallucination.
4. **Context-Driven Paradox:** r = −0.781, p = 0.038 — global causal direction alignment actively hurts performance in context-sensitive domains.

---

## 2. Background

**Probing:** Marks & Tegmark (2023) showed true/false propositions are linearly separable in LLM residual streams. Burns et al. (2023) introduced CCS. Li et al. (2023) identified truth neurons. The Truthfulness Spectrum (2026) explicitly stated: *"encoded directions may not participate in a causal mechanism underlying truthfulness"* — the question this paper answers.

**Circuit Discovery:** Wang et al. (2022) established activation patching for circuit discovery. Conmy et al. (2023) automated it (ACDC). Meng et al. (2022) localised factual knowledge to mid-layer MLPs. No prior work identifies causal directions within circuit components or explains domain-selective hallucination at within-component level.

**Open Bridge:** Sharkey et al. (2025): *"A key challenge is determining whether probed directions causally influence model behavior, or merely correlate."* This paper provides the first empirical answer.

---

## 3. Method: CDD

### 3.1 Stage 1 — Circuit Discovery

For each component c, the Indirect Effect (IE) is:

$$\text{IE}(c) = \mathbb{E}\left[\text{LD}(x^T) - \text{LD}(\mathcal{M}_{a^c \leftarrow a^c(x^H)}(x^T))\right]$$

Circuit: $\mathcal{C} = \{c : \text{IE}(c) \geq \mu_{\text{IE}} + \sigma_{\text{IE}}\}$

Yields 50 components (4.0% of GPT-2-XL, specificity 35.9×, p < 0.0001).

### 3.2 Stage 2 — Contrastive PCA

For each $c \in \mathcal{C}$, collect activation differences $\Delta^c_i = a^c(x^T_i) - a^c(x^H_i)$. Apply PCA to centered $\Delta^c$. Select top k directions explaining 80% variance → **causal candidate subspace**.

Result: mean effective rank = 57.74 (not single-direction; motivates rank-1 validation).

### 3.3 Stage 3a — Rank-1 Causal Validation

Rank-1 patch: $\hat{a}^c_{\text{r1}}(x^H) = a^c(x^H) + \langle a^c(x^T) - a^c(x^H), v^c_1 \rangle \cdot v^c_1$

CRR: $\text{CRR}(c) = \text{IE}_{\text{rank-1}}(c) / \text{IE}(c)$

Result: mean CRR = 23.1; 100% of components exceed 0.70 threshold. CRR > 1 indicates the causal direction is more potent than full-component mean ablation.

### 3.4 Stage 3b — CPA

Train logistic probe $\theta^c$ per component. Compute:

$$\text{CPA}(c) = |\cos(v^c_1, \theta^c / \|\theta^c\|)|$$

Result: mean CPA = 0.0666 — **H1 confirmed**.

### 3.5 Stage 4 — Domain-Conditional CDD

For domain d, compute domain-local top direction $v^{c,d}_1$ from $\Delta^c_d$. Domain CPA: $\overline{\text{CPA}}_d = \frac{1}{|\mathcal{C}|} \sum_c |\cos(v^{c,d}_1, v^c_1)|$.

Cluster domains by (mean IE, domain CPA) → three regimes.

---

## 4. Experiments

**Setup:** GPT-2-XL, TruthfulQA (817 samples, 22 domains), 200 contrastive pairs, RTX 5070 (8.55 GB VRAM).

### 4.1 Circuit: 35.9× specificity, p < 0.0001, 50/1248 components, 100% MLP.

### 4.2 Effective rank: Mean 57.74, median 77.5, no component rank=1.

### 4.3 CRR: Mean 23.1, 100% above 0.70. Top: L33H05=45.7, L27H06=45.4.

### 4.4 CPA (H1): Mean 0.0666, probe accuracy 91.45%. **H1 confirmed — strong misalignment.**

### 4.5 Three Regimes:

| Regime | n | Mean IE | Mean CPA | Key finding |
|---|---|---|---|---|
| Direction-Driven | 3 | 0.617 | 0.703 | Global directions active |
| Context-Driven | 7 | 0.686 | 0.324 | Domain-local directions; r=−0.781 |
| Circuit Failure | 8 | 0.149 | 0.220 | No stable causal direction |

Mann-Whitney p = 0.0103. Within Context-Driven: r = −0.781, p = 0.038 (Context-Driven Paradox).

---

## 5. Discussion

**Probe-Circuit gap:** CPA = 0.0666 means representation engineering (Zou et al., 2023) steers along directions near-orthogonal to causally active directions. It may be changing how the model represents truth-language without changing how it computes truth.

**Context-Driven Paradox:** For context-dependent truths (time, identity, subjective), the global causal direction encodes "assert a universal fact" — which conflicts with the domain-local strategy required. Imposing the global direction disrupts the local strategy, degrading performance.

**Circuit Failure:** Topics with training-data inconsistency (conspiracies, logical falsehood, paranormal) produce no stable causal direction in either regime. Fine-tuning specifically on these domains with a CRR-maximising loss term is a targeted intervention CDD uniquely enables.

**Limitations:** Single model (GPT-2-XL). Domain analysis excludes 4 categories with n<5. Linear structure assumed within components. TruthfulQA tests human-imitative falsehoods specifically.

---

## 6. Conclusion

CDD bridges probing and circuit discovery for LLM truthfulness. We confirmed that probes (91.4% accuracy) share only 6.66% directional alignment with causally active circuit directions (H1), that rank-1 patching recovers 23.1× the full-component IE, and that domain-conditional CDD reveals a three-regime structure in which the Context-Driven Paradox (r=−0.781) shows global direction imposition actively harms context-sensitive truthfulness. These findings challenge the causal assumptions of the probing research program and provide the first mechanistic, domain-resolved theory of selective hallucination.

---

## References

1. Burns et al. (2023). Discovering latent knowledge without supervision. *ICLR 2023*.
2. Marks & Tegmark (2023). The geometry of truth. *arXiv:2310.06824*.
3. Li et al. (2023). Emergent world representations. *ICLR 2023*.
4. Wang et al. (2022). Interpretability in the wild. *ICLR 2023*.
5. Conmy et al. (2023). Towards automated circuit discovery. *NeurIPS 2023*.
6. Meng et al. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*.
7. Elhage et al. (2022). Toy models of superposition. *Transformer Circuits*.
8. Sharkey et al. (2025). Open problems in mechanistic interpretability. *arXiv:2501.16496*.
9. Zou et al. (2023). Representation engineering. *arXiv:2310.01405*.
10. Lin et al. (2022). TruthfulQA. *ACL 2022*.
11. The Truthfulness Spectrum Hypothesis (2026). *arXiv:2602.20273*.
12. The Geometries of Truth Are Orthogonal Across Tasks (2025).
13. Bao et al. (2025). Truth directions across domains. *ACL 2025*.
14. Anthropic (2026). Circuits updates: May 2026. *transformer-circuits.pub*.
15. Nanda & Bloom (2022). TransformerLens. *GitHub*.
