# Issue #11 validation

Executed on 2026-10-15 on an Apple M4 Max, macOS 26.6.2, Python 3.13.13, PyTorch 2.14.1. MPS was built and available; `PYTORCH_ENABLE_MPS_FALLBACK` was unset. See [environment.json](environment.json) for the exact settings, physical memory, CPU configuration and experiment source hashes, and [packages.txt](packages.txt) for the installed package versions.

Setup used `uv venv .venv` and `uv pip install --python .venv/bin/python torch`, which resolved PyTorch 2.14.1. The root `requirements.txt` pins this tested version. No dataset was downloaded.

## Executed checks

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m baseline.run --output evidence/issue-11 --devices auto
.venv/bin/python 01_attention_basics/self_attention_step_by_step.py
uv pip freeze --python .venv/bin/python
```

- **7 CPU tests passed in 3.731 seconds.** [tests.log](tests.log) captures the actual output. Tests cover hand-calculated attention, SDPA and multi-head reference agreement, causal masks and invalid masks, positions/shapes, finite gradients through all parameters, the gradient path from the final prediction to the first position, repeatable CPU training histories, invalid configurations and generated artifact contents.
- **Full baseline completed.** [run.log](run.log) captures command completion. All three architectures trained on CPU for 200 steps at length 32, using seed 7, width 32, batch 32, 8 classes, 256 held-out examples, Adam at 0.003 and gradient clipping at 1.0. [training.json](training.json) preserves all logged loss/accuracy histories, parameter counts and evaluation across lengths.
- **18 benchmark cases completed:** 3 models × 3 lengths (16/32/64) × 2 devices (CPU/MPS). Each case has 3 warmup calls and 10 measured calls for each of forward and train-step latency: **360 recorded timing samples** in [benchmark.json](benchmark.json). MPS calls explicitly synchronize. Memory fields are labeled exact tensor storage, analytical estimates or live MPS allocation snapshots; no peak memory claim is made.
- **The original script ran successfully**, with attention shape `(3, 3)`, row sums `[1, 1, 1]` and output shape `(3, 2)`. See [original-attention.log](original-attention.log). A Git comparison against the starting revision confirms that script is unchanged.
- **Artifact consistency checks passed:** source SHA-256 hashes match the final experiment files; regenerating Markdown from the four saved JSON files exactly matches [report.md](report.md); all benchmark sample counts and positive timings match the configuration. Local links in the root README, baseline guide and report resolve.
- **Whitespace validation:** `git diff --cached --check` passed for the staged implementation, documentation and evidence.

PyTorch emitted an optional-NumPy initialization warning, preserved in the logs. NumPy is not installed or used; all commands and tests above succeeded without it.

## Evidence and limits

At the training length, final held-out accuracy was RNN **8.2%**, LSTM **9.8%**, and Transformer **100%**, against **12.5%** chance. The recurrent models did not learn this dependency with this budget. This is a single-seed synthetic comparison with equal widths and step counts, not equal parameter counts or tuned architecture comparisons. The report retains these outcomes.

Timings are visibly noisy and non-monotonic: the CPU Transformer forward median at length 32 exceeded that at length 64, and MPS Transformer forward medians decreased between length 16 and 64. The saved min/max and individual samples show substantial within-run variability. These short timings do not isolate the contributions of host contention, launch overhead, compilation or thermal effects. No general speed ranking, quadratic empirical scaling fit or hardware-wide conclusion follows from them. The quadratic attention storage values are analytical estimates.

Full MPS learning comparisons, larger training budgets, longer sequences and independent repeated hardware trials were **not run**. The generated report separates recorded values from structural theory and those optional follow-ups. This integrated baseline does not replace the fuller acceptance criteria of stage Issues #2–#8. Issue #11 requires no post-merge verification.
