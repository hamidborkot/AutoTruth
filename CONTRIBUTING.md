# Contributing to AutoTruth

Thank you for your interest in contributing! Here are the guidelines:

## Ways to Contribute

1. **Cross-model validation** — Run AutoTruth on Mistral-7B, Llama-2-7B, Qwen2-7B, Gemma-7B and submit results
2. **Cross-dataset extension** — Apply to GSM8K (math), HumanEval (code), CommonsenseQA
3. **Bug reports** — Open an issue with your error log and hardware specs
4. **Performance improvements** — The patching loop (Cell 9) is the bottleneck; batched patching would reduce time significantly

## Pull Request Process

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Add your results to `results/` with a descriptive filename including model name and date
4. Update `REPRODUCIBILITY.md` if you add a new experiment
5. Submit PR with a clear description of what you ran and what you found

## Cross-Model Results Template

If you run on a new model, please report:
```json
{
  "model": "<model_name>",
  "n_samples": <int>,
  "specificity_ratio": <float>,
  "p_value": "<string>",
  "circuit_size": <int>,
  "total_components": <int>,
  "sparsity_pct": <float>,
  "top_5_circuit": ["..."],
  "hardware": "<GPU model>",
  "runtime_hours": <float>
}
```

## Code Style

- Python 3.10+, PEP 8
- All functions must have docstrings
- All random operations must use `SEED = 42`
- No hardcoded paths — use `pathlib.Path`
