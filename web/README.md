# Browser-local recall Transformer

This small static app runs the **actual CLI-trained, one-block custom Transformer** from [baseline/models.py](../baseline/models.py) through ONNX Runtime Web's WebGPU execution provider. It predicts the first token using the final position; it is not a chatbot. There is no inference backend or browser training.

## Train, export, serve

From the repository root, using Python 3.11+ supported by the pinned PyTorch release (the optional NumPy export dependency requires 3.11+) and Node.js 20+ with npm:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-export.txt
.venv/bin/python -m baseline.run --devices cpu --steps 200 \
  --output results/webgpu --checkpoint results/webgpu/transformer.pt
.venv/bin/python -m baseline.export --checkpoint results/webgpu/transformer.pt \
  --output web/model --lengths 2 16 32 64
npm --prefix web ci
npm --prefix web run build
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open **http://localhost:8000** in a desktop browser with working WebGPU and hardware acceleration. The server only serves static files. All runtime/model assets are served from the same origin; there are no CDN imports, remote inference calls, accounts or API keys. npm/Python dependency installation requires network access during setup. Closing the server stops file serving, not a Python model process.

The original CLI command and its JSON/report artifacts remain available without export dependencies or `--checkpoint`. `--checkpoint` saves only the final trained Transformer, with model dimensions, positional capacity, training settings and a versioned state dict; it does not save optimizer state or claim resumable training. `baseline.checkpoint.load_transformer` reloads it on CPU using `weights_only=True`. Keep your checkpoint to export again without retraining. Generated checkpoints, model bundles, runtime assets and test outputs are ignored by Git.

The app initializes WebGPU, loads the first supported length and warms it twice. Enter valid IDs, run inference, then select a head to inspect attention from that exact inference. Change length to load a separate export of the same weights. The prediction can be wrong: a short training run and evaluation at other lengths do not guarantee successful recall. The saved training report records actual accuracy.

## Model and input contract

- Batch size 1, float32 computations, int32 input IDs. Default vocabulary is the eight strings `0` through `7`, mapped identically to classes. No free text, tokenizer, padding or causal mask; attention is bidirectional.
- Lengths are explicitly listed in `model/manifest.json`. Export accepts lengths from 2 through the smaller of 128 and the checkpoint's positional capacity. The default CLI capacity is 64. For 128, train with `--lengths 16 32 64 128`, then export that length. No interpolation or resizing of learned weights occurs.
- Every export has fixed batch and length. This lets ONNX fold shape calculations instead of assigning them to CPU. A separate file per length duplicates this tiny model's parameters and keeps this first version inspectable. Unexported lengths are rejected.
- The checkpoint is the source of truth for settings and weights. `manifest.json` records the vocabulary, dimensions, training configuration, parameter bytes, artifact sizes/hashes and export versions; `reference.json` holds fixed PyTorch inputs/outputs. `recall-L.onnx` returns `logits [1, classes]` and `attention [1, heads, L, L]` in one execution.
- Attention remains explicit learned Q/K/V, scaled matrix multiplication and softmax. A small optional return path exposes the very weights used for mixing values in the block, with no architecture replacement or second inference. Logits match the normal CLI forward exactly. Rows are queries and columns are keys; labels contain both position and token ID. The final query feeds the classifier after residuals, feed-forward and normalization. Attention alone does not establish feature importance or causality.

The exporter intentionally uses PyTorch's legacy tracer (`dynamo=False`, ONNX opset 17) to preserve the small explicit graph. PyTorch emits deprecation and shape-constant tracing warnings. Length and head width really are fixed for each artifact; the exporter validates their limits and checks every supported shape numerically. This is not a dynamic-length export or support for arbitrary models.

## WebGPU status and metrics

Use a current Chrome or Edge desktop installation with a supported adapter as the first target. WebGPU also depends on the OS, hardware, driver and browser configuration; a browser name alone does not guarantee availability. HTTPS or localhost is required; opening `index.html` with `file://` is unsupported. Other browsers/hardware are optional validation. See the upstream [WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) for prerequisites and API details.

The app checks for a GPU adapter, uses the device initialized by the pinned ONNX Runtime Web **1.22.0** WebGPU provider, and sets `session.disable_cpu_ep_fallback=1`. An unsupported operator/device or missing artifact fails visibly. It never retries with a CPU/WASM execution provider. WASM still orchestrates the ONNX session; that is separate from model compute falling back to CPU. Adapter absence and startup failures leave inference disabled. Device loss is reported on subsequent runs; reload to initialize again.

Model fetch/session initialization and two warmup calls are reported separately. The displayed inference metric is a **single warmed, synchronized WebGPU run + readback wall time**, measured with `performance.now()` around `session.run()` and `device.queue.onSubmittedWorkDone()`. It includes dispatch, transfers and synchronization, and excludes rendering and model loading. It is not kernel-only GPU time (not measured), a throughput benchmark, or a CPU/GPU speed comparison. Reloading a length releases the previous session; the runtime may retain internal caches.

Memory displays include ONNX file bytes, float32 parameter storage, `heads * length² * 4` bytes for one attention output, input bytes and logits bytes. These are exact logical sizes/analytical estimates, **not measured allocations or peak GPU memory**. They exclude positions, other activations/copies, runtime/workspace and allocator overhead. Browser JS heap or unified-memory capacity cannot establish GPU peak usage. No GPU-memory profiling is claimed.

## Reproducible checks

Generate the model and runtime assets above first, then run:

```sh
.venv/bin/python -m unittest discover -s tests -v
npm --prefix web test
npm --prefix web run build
npm --prefix web run test:browser
```

The browser test uses Playwright with an **installed Google Chrome** (`channel: 'chrome'`) and its real default WebGPU device, with no software GPU or unsafe-WebGPU flags. It starts/stops a localhost static server on port 18765. Stop any conflicting server or change the test's config and origin together. No browser download is required with installed Chrome. `web/test-results/` contains a screenshot and machine-readable `webgpu.json` on a successful integration run.

Python checks prove checkpoint reload uses trained weights, attention comes from the same forward, CLI logits are unchanged, and each fixed ONNX artifact passes ONNX checking and native CPU ONNX Runtime comparisons (`atol=1e-5`, `rtol=1e-5`, exact predicted class). Export failure does not publish a new manifest.

The app's **Compare with PyTorch fixtures** action runs three deterministic inputs at every exported length: cycling all IDs, reverse cycling, and repeated zeros. Both logits and every per-head attention weight must satisfy `abs(actual - expected) <= 1e-4 + 1e-3 * abs(expected)` and predicted class must match exactly. It reports measured maximum absolute errors and run wall times. These float32 browser tolerances accommodate device arithmetic; they do not waive a prediction mismatch.

The Playwright integration exercises that same verification action, input errors, every exported length, head switching and labeled heatmaps. It blocks off-origin requests and checks for uncaught page errors. Separate tests remove WebGPU to check the visible unavailable state and supply a valid CPU-only ONNX operator to verify that CPU fallback is rejected. If Chrome has no adapter, the GPU integration is explicitly skipped with a saved capability probe; that is a verified environment limitation, **not a GPU pass**. Installation, operator or numerical failures on a capable adapter fail the test.

Actual results and environment are in [the Issue #14 validation record](../evidence/issue-14/validation.md). Do not infer successful GPU execution from passing Python or pure-JavaScript checks.

## Static hosting

After training/export and `npm ci` / `npm run build`, upload `index.html`, `style.css`, `app.js`, `core.js`, `runtime.js`, the full `vendor/` directory the full `model/` directory and `THIRD_PARTY_NOTICES.txt` together. Preserve paths, MIME types (`.mjs` as JavaScript, `.wasm` as `application/wasm`) and serve over HTTPS. The single-threaded runtime does not need cross-origin-isolation headers. All paths are relative, including WASM/model URLs, so hosting under a repository subpath works.

For GitHub Pages, put those files at the root of a Pages deployment artifact or the chosen Pages branch/folder and enable Pages for that source. Include `.nojekyll` if publishing a branch. Generate ignored model/vendor assets before packaging; source checkout alone is not a complete deployment. Only public static assets belong in the bundle; the Python checkpoint is unnecessary for serving. No deployment workflow or publication is included here. There are no required post-merge checks.
