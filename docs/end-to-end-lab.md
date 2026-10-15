# End-to-end Stage 0–7 baseline

This small, educational experiment complements the original step-by-step attention script. It is a **working baseline**, not a reproduction of the large-scale results in *Attention Is All You Need*.

## Install and run

Python 3.10+ and PyTorch are required. Install PyTorch using instructions for your machine at https://pytorch.org/get-started/locally/ .

From the repository root:

    python3 -m unittest -v test_lab
    python3 lab.py --output results

For a very fast smoke run:

    python3 lab.py --output results-smoke --epochs 1 --steps 2 --bench-lengths 4 8 --warmup 1 --repeats 2

On an Apple Silicon Mac, when PyTorch MPS is available:

    python3 lab.py --device both --output results-mac --epochs 5 --steps 100 --bench-lengths 8 16 32 64 128 --repeats 10

Use the CPU baseline to compare experiments; MPS performance depends on the device, driver and PyTorch version. The run deliberately errors if MPS is requested but unavailable.

## Outputs

The specified output directory contains:

- environment.json — versions, CPU metadata, available MPS backend, arguments, random seed (Stage 0)
- attention.json — input, Q/K/V, attention scores, scaling, weights, row sums and output; compared numerically with PyTorch (Stages 1–2)
- training.csv — held-out accuracy, loss, and parameter count for an RNN, LSTM and manually assembled Transformer on an identical synthetic first-bit recall task (Stages 3–5)
- benchmark.csv — repeated/warmed forward and complete training-step wall-clock timing at multiple lengths (Stage 6)
- synthesis.md — report generated from actual training and benchmark CSV rows, with theoretical expectations and limitations explicitly separated (Stage 7)

The Transformer implementation in lab.py uses separate Q/K/V projections, multi-head attention, sinusoidal positional encoding, residual connections, layer normalization and a feed-forward block. It does not use nn.Transformer. In addition to the numerical attention experiment, test_lab.py checks mask behavior, tensor shapes, trainable gradients and all three model types.

## Interpretation and caveats

- Attention weight matrices require quadratic space with increasing length **when materialized explicitly**. The byte count in benchmark.csv is a **theoretical fp32 matrix estimate**; it is not observed peak RSS/GPU memory, nor the full memory consumption of the model. Backend kernels may avoid materializing some matrices.
- Timings include backend/kernel and optimizer overhead. The training-step timings include forward, backward and optimizer updates; inference timings do not. MPS synchronizes before stopping the clock.
- Models share data, seed and optimizer family, but differ in parameter count. Validation uses a fixed held-out pseudo-random sample, and the learning task tests only one kind of long-range dependency. No claim that one model wins in general follows from this experiment.
- Stage 7 is **automatically generated** from the data collected on the machine running the experiment. CPU data cannot be presented as Apple MPS measurements.
- More seeds, larger sequence lengths, real measured memory instrumentation and stronger evaluation are worthwhile follow-ups in the original individual stage Issues (#2–#8).
- Original hand-inspection experiment remains unchanged at 01_attention_basics/self_attention_step_by_step.py.

Umbrella implementation issue: #11.
