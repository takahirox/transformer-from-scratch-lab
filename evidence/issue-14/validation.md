# Issue #14 validation

Implemented and checked locally on 2026-10-20 JST (2026-10-19 UTC). No deployment, push, PR, comment or merge was performed. No required post-merge verification.

Environment: Apple M4 Max, macOS 26.6.2 arm64, Python 3.13.13, PyTorch 2.14.1, NumPy 2.4.2, ONNX 1.20.1, native ONNX Runtime 1.24.2, Node 24.12.0, npm 11.6.2, ONNX Runtime Web **1.22.0**, Playwright 1.58.2 and installed Google Chrome 155.0.8059.80. The browser reported a non-fallback Apple `metal-3` WebGPU adapter. Its compatibility user-agent string reports Intel/macOS 10.15.7; the actual host environment is recorded in [environment.json](environment.json).

The source Issue is [#14](https://github.com/takahirox/transformer-from-scratch-lab/issues/14). This work preserves the explicit one-block classifier and CLI comparisons, adds an opt-in trained checkpoint, fixed-shape exports with attention from the same forward, and a static client-only WebGPU demo. Training is CPU-only; browser training is not implemented.

## Commands and actual results

Commands ran from the repository root. Python dependencies were installed with `uv venv .venv` and `uv pip install --python .venv/bin/python -r requirements.txt onnx==1.20.1 onnxruntime==1.24.2 numpy==2.4.2`. Web dependencies were installed with npm and the final lockfile was verified with `npm --prefix web ci` ([log](npm-ci.log)).

```sh
.venv/bin/python -m baseline.run --devices cpu --output results/webgpu \
  --checkpoint results/webgpu/transformer.pt --steps 200
.venv/bin/python -m baseline.export --checkpoint results/webgpu/transformer.pt \
  --output web/model --lengths 2 16 32 64
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python 01_attention_basics/self_attention_step_by_step.py
npm --prefix web run build
npm --prefix web test
npm --prefix web run test:browser
git diff --check
```

- The full existing CLI training/benchmark command completed on CPU and saved a reusable trained Transformer checkpoint. Seed 7, width 32, four heads, training length 32, 200 Adam steps; final held-out evaluation loss **0.0089382293**, accuracy **1.0**. [Transformer training record](transformer-training.json) includes the history and evaluations at other lengths. This is one synthetic experiment, not an accuracy guarantee.
- Export completed for lengths **2, 16, 32, 64** ([log](export.log)). Every graph passed `onnx.checker` and native ONNX Runtime comparisons against PyTorch on three deterministic inputs per length. Maximum absolute errors: logits **1.5497208e-6**, attention **5.3644180e-7**; `atol=rtol=1e-5`, all predictions identical. [Native comparison record](native-onnx.json) retains inputs, logits and errors; [manifest](manifest.json) retains model/settings/vocabulary and checkpoint/model hashes. Full attention reference arrays remain in the generated `web/model/reference.json` and are reproducible by exporting again.
- **9 Python tests passed**, none skipped ([log](python-tests.log)), including all seven original tests, actual trained-checkpoint reload, same-forward attention/CLI-logit equivalence, odd-width export, numerical ONNX comparisons and unsupported length rejection. Expected legacy-export deprecation and shape-constant tracer warnings are retained in the logs; these exports intentionally have fixed batch, length and head width.
- The original attention script completed ([log](original-attention.log)); attention shape `(3, 3)`, row sums `[1, 1, 1]`, output shape `(3, 2)` and expected hand values were inspected.
- Runtime asset packaging passed ([log](web-build.log)); assets are served locally with no CDN. **3 JavaScript tests passed** ([log](js-tests.log)): valid/invalid inputs, numeric comparison failures and tensor-size estimates.
- **3 Playwright browser tests passed**, none skipped ([log](browser-tests.log)). Headless installed Chrome used its real default WebGPU adapter, without software-GPU or unsafe-WebGPU flags. All **12 PyTorch reference cases** passed for predictions, logits and every attention element with `atol=1e-4`, `rtol=1e-3` and exact class agreement. Maximum absolute browser errors: logits **2.0861626e-6**, attention **6.5565109e-7**. [Browser evidence](browser-webgpu.json) records the adapter, user agent, each fixed input, errors, timing samples and observed request origins. Requests were restricted to localhost, with no remote inference or off-origin asset requests. Input errors, each exported length, head switching and heatmap rows were exercised; no uncaught page errors occurred.
- Browser negative checks passed: removing `navigator.gpu` produced a visible error and disabled inference. Supplying a valid ONNX `Unique` graph (CPU-only in this runtime) produced the runtime diagnostic **“fallback to CPU EP has been explicitly disabled”** and disabled inference. This checks actual rejection of CPU fallback, not just the provider option in source code.
- A full-page screenshot from the browser test was visually inspected: predictions, token/axis labels, per-head heatmap, timing and memory notes were visible. It is generated at `web/test-results/.../demo.png`, not included as a source asset.
- Whitespace validation passed with `git diff --check` (also checked after staging new files).

Logs replace absolute worktree prefixes with `<worktree>` and remove terminal color escapes; result values are unchanged. Generated weights, ONNX bundles, runtime binaries and test screenshots are intentionally ignored. Follow [web/README.md](../../web/README.md) to regenerate them in a fresh checkout.

## Runtime selection and limits

Early browser checks exposed packaging and runtime integration failures, which were fixed before the passing run. The initially selected ONNX Runtime Web 1.24.2 package required Asyncify assets; its native WebGPU backend failed a run with a custom device and destroyed the default device on session release, preventing length changes. The final implementation pins **1.22.0**, whose JSEP WebGPU backend passed session replacement, numerical and CPU-fallback rejection tests. Python's native runtime remains 1.24.2 for export verification. A future browser-runtime upgrade must rerun these integration checks; matching package version numbers does not establish compatibility.

Displayed timing is a warmed, synchronized **WebGPU run + readback wall time** and is separate from fetch/session initialization and two warmup runs. It includes CPU dispatch and transfers; GPU kernel-only duration was not measured. Browser timing samples are evidence of this tiny run, not a benchmark or speedup claim. Parameter/file sizes and attention/output/input storage estimates are labeled; peak GPU allocation was not measured.

Only this Chrome/Apple adapter was tested. Edge, Safari, Firefox, other GPU vendors, browser length 128, GPU memory profiling and static deployment were not tested. HTTPS/static-hosting instructions are supplied; no publication is claimed.
