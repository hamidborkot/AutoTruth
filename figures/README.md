# Figures Directory

All publication-quality figures (300 DPI PNG) for the AutoTruth paper.

## Files

| File | Description | Used In |
|---|---|---|
| `autotruth_main_figure.png` | 4-panel main result (component scores, layer dist, ablation, cross-domain) | Paper Figure 2 |
| `autotruth_circuit_heatmap_final.png` | Layer × Head heatmap + MLP bar chart | Paper Figure 3 |

## Regenerating Figures

```bash
jupyter nbconvert --to notebook --execute notebooks/09_visualizations.ipynb
```

Or run `Final Cell A` and `Final Cell C` from the notebook directly.
