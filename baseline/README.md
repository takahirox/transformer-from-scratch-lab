# Integrated Stage 0–7 baseline

This is a small executable baseline for [Issue #11](https://github.com/takahirox/transformer-from-scratch-lab/issues/11), alongside the unchanged [original attention script](../01_attention_basics/self_attention_step_by_step.py). It does not replace the richer stage Issues #2–#8.

## Setup and commands

Use Python 3.10+ on a platform supported by the pinned PyTorch release. The recorded run uses Python 3.13. No dataset downloads, training framework, GPU, or test runner dependency is needed. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m baseline.run --devices cpu --output results/cpu
```

With `uv`, use `uv venv .venv` and `uv pip install --python .venv/bin/python -r requirements.txt` for setup instead. PyTorch may warn that NumPy is absent; these experiments use no NumPy APIs and work without it.

To also benchmark MPS **when actually available**:

```sh
.venv/bin/python -m baseline.run --devices auto --output results/auto
```

CPU is always benchmarked. If MPS is unavailable, `auto` records that it was not run. Device errors fail the command rather than fabricate results. Do not enable `PYTORCH_ENABLE_MPS_FALLBACK=1` for a device comparison; the environment artifact captures this setting. Timing calls synchronize MPS before and after every sample; transfers to the device are outside the timed region. MPS training-step benchmarks update fresh model weights, but are not full MPS learning experiments.

The default run trains three small models for 200 steps and benchmarks lengths 16, 32, and 64 with 3 warmup calls and 10 samples per operation. For a longer optional experiment:

```sh
.venv/bin/python -m baseline.run --steps 1000 --length 64 \
  --lengths 16 32 64 128 --warmup 10 --repeats 30 \
  --devices auto --output results/longer
```

Run `--help` for options. `Config` in [run.py](run.py) contains all settings, including thread count, dimensions, batch size, learning rate and evaluation size. Outputs are overwritten in the selected directory; use a new output directory to preserve each experiment. The `results/` directory is ignored. A completed run prints its output location and creates `report.md`; an interrupted run can leave partial JSON evidence.

## What each stage does

| Stage | Implementation | Saved evidence |
| --- | --- | --- |
| 0 | Seeds, float32, CPU thread count, deterministic algorithm settings; Python/PyTorch/OS/CPU/physical memory/MPS metadata and source hashes | `environment.json` |
| 1–2 | Explicit X/W/Q/K/V, scores, scale, softmax, weighted values and tensor shapes; numerical check against PyTorch SDPA | `attention.json` |
| 3–4 | Learned per-head Q/K/V projections, concatenation/output projection, fixed sinusoidal positions, pre-normalization, residuals and feed-forward network; one trainable Transformer block | [models.py](models.py), training results |
| 5 | RNN/LSTM/Transformer on shared first-token recall data; train loss and held-out loss/accuracy histories, parameter counts, held-out accuracy at other lengths | `training.json` |
| 6 | Warmed, repeated forward and Adam train-step wall timing at multiple lengths; analytical storage estimates and conditional MPS allocation snapshots | `benchmark.json` |
| 7 | Markdown tables generated exclusively from saved values; measurements, structural theory, limitations and unrun cases | `report.md` |

The attention mask convention is **True = allowed**. Masks broadcast to `(batch, heads, query_length, key_length)`; each query must have an allowed key. Tests include causal masking and rejection of fully masked queries. The recall classifier uses bidirectional attention and fixed-length examples, so training needs no causal or padding mask. It predicts the first token from the final position, with independent random distractors between them; the label never appears as a special input field. This isolates information transport rather than language modeling.

## Interpreting comparisons

All architectures use the same width, batch size, training steps, optimizer, learning rate, clipping and exact minibatch stream. Separate fixed seeds generate held-out examples. Parameter counts differ and are recorded. Each model has one recurrent layer or one Transformer block; no dropout, pretrained weights or hyperparameter tuning is used. Evaluation across lengths uses the same trained weights, including lengths not seen during training. It is not training at each length or proof of generalization to arbitrarily long inputs.

CPU seeds and deterministic settings make training records repeatable in the same software environment; tests check this. Versions, device kernels and timings can differ between machines. Benchmark cases use fresh, untrained models and fixed device-resident inputs. Their warmup and repeated train steps mutate weights; forward measurements use eval/no-grad. Raw samples and min/median/max are saved. Short runs can be dominated by launch overhead, compilation, CPU frequency changes or thermal effects; do not infer asymptotic scaling from a few short lengths.

Memory fields explicitly distinguish exact parameter tensor storage, analytical gradient/Adam-moment/one-attention-matrix estimates, and optional live MPS tensor allocation snapshots. None is a measured process or activation peak. One attention matrix scales as `batch * heads * length² * sizeof(float32)` and excludes the softmax copy, saved activations, workspaces and allocator overhead. CPU allocator memory is not measured. Do not compare these estimates to unified-memory capacity as a complete model footprint.

## Recorded validation

The checked-in [report](../evidence/issue-11/report.md) and [validation record](../evidence/issue-11/validation.md) identify the actual host, settings, commands, outputs and limitations. Generate your own results before making claims about another machine. The CPU tests cover hand values, reference agreement, multi-head shape/projection equivalence, causal masking, positional encodings, gradients through all models, repeatable training, invalid settings and end-to-end artifact contents. There are no required post-merge checks for Issue #11.
