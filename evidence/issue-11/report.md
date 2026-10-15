# Stage 0–7 baseline synthesis

Generated from [environment.json](environment.json), [attention.json](attention.json), [training.json](training.json), and [benchmark.json](benchmark.json).

## Measurements

Environment: Python 3.13.13, PyTorch 2.14.1; macOS-26.6.2-arm64-arm-64bit-Mach-O; Apple M4 Max. CPU threads: 1; seed: 7; float32. MPS available: True. MPS status: measured. PYTORCH_ENABLE_MPS_FALLBACK: unset.

Stages 1–2: explicit projections, raw/scaled scores, weights, shapes and weighted values are in attention.json. Max absolute error against PyTorch SDPA: 1.19e-07; weight row sums: [1.0, 1.0, 1.0].

Stages 3–4: the trained Transformer uses learned Q/K/V and output projections, split/concatenated heads, fixed sinusoidal positions, pre-layer normalization, residual connections and a two-linear-layer ReLU feed-forward network.

Stage 5: first-token recall from the final position, 8 classes (chance 12.5%), length 32, 200 Adam steps, batch 32, learning rate 0.003. All models receive identical minibatches and held-out examples; widths and step budgets match, parameter counts differ. Training is on CPU.

| Model | Parameters | Initial eval loss | Final train loss | Final eval loss | Final eval accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| rnn | 2632 | 2.1451 | 2.0993 | 2.1063 | 8.2% |
| lstm | 8968 | 2.0928 | 2.0839 | 2.0905 | 9.8% |
| transformer | 9128 | 2.2306 | 0.0103 | 0.0089 | 100.0% |

Observed held-out loss changes (final minus initial; negative means improvement): rnn -0.0389; lstm -0.0023; transformer -2.2217. These are outcomes of this budget and seed, not architecture-wide rankings.

Held-out accuracy by dependency distance (same trained weights; no retraining):

| Model | Length | Accuracy |
| --- | ---: | ---: |
| rnn | 16 | 7.4% |
| rnn | 32 | 13.3% |
| rnn | 64 | 11.3% |
| lstm | 16 | 8.2% |
| lstm | 32 | 13.7% |
| lstm | 64 | 10.9% |
| transformer | 16 | 100.0% |
| transformer | 32 | 100.0% |
| transformer | 64 | 100.0% |

Stage 6: 3 warmup calls and 10 timed samples per operation. perf_counter wall time; per-sample device synchronization; inputs already on device; eval/no_grad forward; train step includes zero_grad, cross entropy, backward, clipping, Adam; fresh untrained model per case, reused/mutated through warmup and samples.

| Device | Model | Length | Forward median (ms) | Train-step median (ms) | One attention matrix estimate (bytes) |
| --- | --- | ---: | ---: | ---: | ---: |
| cpu | rnn | 16 | 0.314 | 2.588 | 0 |
| cpu | rnn | 32 | 0.635 | 2.558 | 0 |
| cpu | rnn | 64 | 1.117 | 3.481 | 0 |
| cpu | lstm | 16 | 0.928 | 3.011 | 0 |
| cpu | lstm | 32 | 1.306 | 6.143 | 0 |
| cpu | lstm | 64 | 3.450 | 14.218 | 0 |
| cpu | transformer | 16 | 1.385 | 27.669 | 131072 |
| cpu | transformer | 32 | 16.417 | 3.578 | 524288 |
| cpu | transformer | 64 | 2.421 | 5.725 | 2097152 |
| mps | rnn | 16 | 15.332 | 25.614 | 0 |
| mps | rnn | 32 | 13.503 | 58.005 | 0 |
| mps | rnn | 64 | 22.599 | 64.599 | 0 |
| mps | lstm | 16 | 5.004 | 17.856 | 0 |
| mps | lstm | 32 | 11.715 | 10.843 | 0 |
| mps | lstm | 64 | 9.760 | 20.828 | 0 |
| mps | transformer | 16 | 11.100 | 24.048 | 131072 |
| mps | transformer | 32 | 5.554 | 19.976 | 524288 |
| mps | transformer | 64 | 7.314 | 23.412 | 2097152 |

Observed forward median ratios, length 64 / 16:

- cpu rnn: 3.56x.
- cpu lstm: 3.72x.
- cpu transformer: 1.75x.
- mps rnn: 1.47x.
- mps lstm: 1.95x.
- mps transformer: 0.66x.

Exact parameter storage and analytical tensor estimates, NOT peak/process memory. One attention matrix is B*H*L^2*4 bytes; excludes other activations, gradients, allocator/workspace overhead. MPS snapshot is live tensor allocation after train steps, not a peak; CPU allocator memory is not measured.

## Theory and interpretation

Dense attention forms a score for every query/key pair: O(L²D) work and O(HL²) score storage per batch item. Recurrent layers take L sequential transitions with O(LD²) work. Attention directly connects distant positions within a layer; recurrent models carry information through intermediate states. These are structural claims, not measured asymptotic rates or guarantees of better accuracy.

The tables show this run's learning and timing behavior. Any architecture below chance or failing to learn is retained as evidence of this small budget and setup. Final-position recall tests information transport across distractors, not language modeling. Timing rankings at these short lengths cannot establish a universal speed advantage.

## Limitations and unrun cases

One seed, one synthetic task, one layer, small widths and budgets; no hyperparameter search, matched parameter budgets, confidence intervals or large-length scaling claim. Repeated timing samples describe within-run variability, not independent experiments. CPU training uses deterministic algorithms; warn-only mode records the possibility of nondeterministic device kernels. Exact cross-version/device reproducibility is not promised. Evaluation at other lengths is generalization, not a comparison of models trained at those lengths.

MPS: measured. No MPS training comparison was performed. Larger training, longer sequences and repeated Apple Silicon CPU/MPS trials remain optional follow-up experiments. Memory estimates are not measured peaks. This integrated baseline does not complete the richer stage Issues #2–#8.
