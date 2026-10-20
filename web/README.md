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

After training/export and `npm ci` / `npm run build`, the app can be served as static files over HTTPS. Preserve paths and MIME types (`.mjs` as JavaScript, `.wasm` as `application/wasm`). The single-threaded runtime does not need cross-origin-isolation headers. All paths are relative, including WASM/model URLs, so hosting under a repository subpath works. `npm run build` also downloads ONNX Runtime 1.22.0's upstream third-party notices from a fixed source commit and verifies their SHA-256; setup needs access to `raw.githubusercontent.com` as well as the dependency registries.

## GitHub Pages build and deployment

The expected public URL is **https://takahirox.github.io/transformer-from-scratch-lab/**. Publication is handled by [`.github/workflows/pages.yml`](../.github/workflows/pages.yml), following GitHub's [custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) contract. A successful local or PR build does not establish that this URL is published.

One-time repository administrator setup:

1. In **Settings → Pages → Build and deployment → Source**, select **GitHub Actions**. Do not select a publishing branch or commit the generated assets.
2. Ensure repository Actions policies allow the `actions/*` actions in the workflow and deployment tokens with `pages: write` and `id-token: write`. The `github-pages` deployment environment must permit `main`; retain any intentional environment protection rules.
3. After merge, inspect the actual publication run. The workflow checks Pages configuration without trying to enable Pages automatically. If admin configuration blocks publication, record the exact required setting and track the blocker in a new `task` Issue as described below.

Pull requests targeting `main` run the build/check job when the workflow, Python model/tests/dependencies, or web files change. These runs have only `contents: read` and never upload a Pages deployment artifact or deploy. Relevant pushes to `main` build the same artifact and publish it; manual **Actions → WebGPU demo Pages → Run workflow** also publishes when the selected branch is `main`. Dispatches from other branches only validate. Concurrency separates PRs and serializes production runs without cancelling an active deployment. Only the dependent deployment job receives Pages/OIDC write permissions and uses the `github-pages` environment.

The build uses Ubuntu 24.04, Python 3.13.13, Node 24.12.0, pinned numerical dependencies from `requirements-export.txt`, a CPU PyTorch wheel, and `npm ci` with the checked-in lockfile. It trains the actual classifier with seed 7, one CPU thread, deterministic algorithms, 200 Adam steps, training length 32 and positional capacity 64. The CLI still runs its RNN/LSTM comparisons; benchmark warmup/repeats are reduced to one each for this deployment build. No external dataset or checkpoint cache is needed. The manifest records the complete model/training settings, tool versions, checkpoint hash, and model hashes. Repeatability is for the same tools/platform; bit-identical results across different OSes, hardware or dependency versions are not promised. Timings and environment timestamps naturally vary.

Local reproduction from the repository root (create `.venv` and install `requirements-export.txt` as above; on Linux the workflow installs `torch==2.14.1` from `https://download.pytorch.org/whl/cpu` first):

```sh
npm --prefix web ci
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python 01_attention_basics/self_attention_step_by_step.py
npm --prefix web test
.venv/bin/python -m baseline.run --devices cpu --seed 7 --steps 200 \
  --length 32 --lengths 16 32 64 --warmup 1 --repeats 1 \
  --output results/pages-training --checkpoint results/pages-training/transformer.pt
.venv/bin/python -m baseline.export --checkpoint results/pages-training/transformer.pt \
  --output web/model --lengths 2 16 32 64
npm --prefix web run build:pages
npm --prefix web run test:pages
python3 -m http.server 8000 --bind 127.0.0.1 --directory results/pages
```

Open **http://localhost:8000/transformer-from-scratch-lab/**. The publishable artifact is only `results/pages/transformer-from-scratch-lab/`; its contents become the site root under that repository subpath. The packager copies an explicit allowlist: HTML/CSS, three app JS modules, four ONNX models, manifest/reference JSON, three runtime JS/WASM files, and runtime license/notices. It validates nonempty regular files, literal source asset references, model sizes/hashes and reference coverage/shapes. It rejects duplicate/unsafe model paths and sites above **50 MiB**, comfortably below the [Pages 1 GB site limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits), and prints the actual total bytes including runtime and references. Checkpoints, Python environments, training output, source tests, credentials and `node_modules` are excluded. Generated bundles remain gitignored.

Pre-merge validation includes the existing Python/JS tests, ONNX graph checking and native reference comparisons during export, negative packaging checks, and HTTP path/MIME/hash checks against the assembled artifact under `/transformer-from-scratch-lab/`. `test:pages` also reuses the existing Chrome WebGPU integration tests at that subpath and checks the visible unsupported state. It requires installed Chrome (CI installs it via Playwright). Without an adapter, GPU-dependent tests explicitly skip; the capability probe and test results are retained in the `pages-validation` Actions artifact along with training/environment reports and the export manifest. A skipped GPU test is **unverified WebGPU execution**, not a pass. Browser installation, packaging, operator or numerical failures on a capable adapter fail the build. Use Chrome/Edge with a working WebGPU adapter and hardware acceleration over HTTPS or localhost; no inference server or CPU fallback is introduced.

The implementation's recorded pre-merge results are in [the Issue #16 validation record](../evidence/issue-16/validation.md). Review the diff and actual evidence before merge; do not require a live site before merging.

## Required post-merge verification (pending until publication)

This is separate from pre-merge approval. After merging the reviewed implementation:

1. Inspect the **actual** `WebGPU demo Pages` Actions run triggered on `main`, wait for completion, and inspect build/deploy logs and the environment's deployed URL. Record the workflow-run URL and outcome; a green PR run does not prove deployment succeeded.
2. On success, request the published HTTPS URL above. Verify the page, JS/CSS, `model/manifest.json`, its `reference.json` and every ONNX model, plus the vendor JS/WASM and notices under the repository subpath. Check HTTP status/MIME types, model hashes and the app's initialization or clear unsupported-WebGPU message. On a capable browser/device, run **Compare with PyTorch fixtures** and record the browser/adapter and result; otherwise record real-WebGPU inference as unverified with the concrete limitation.
3. Save the run URL, publication URL, HTTP/browser results and device limitations separately from the pre-merge decision. Publication remains **pending** until this check is actually performed.
4. If deployment fails, the site is broken, or an admin setting blocks publication, create a **new `task`-labeled follow-up Issue** linking Issue #16 and the merged PR. Include the deployment-run URL, observed errors/symptoms, expected behavior, reproduction, exact admin action if needed, and actionable pre-merge acceptance criteria. Fix it in a separate PR. Do not retroactively fail the original pre-merge validation or hold the original merge for a deployment that requires that merge.

Issue #16 may close after its complete implementation and pre-merge checks pass and it is merged; required post-merge verification remains explicitly pending until verified. Additional physical browsers/GPUs and caching/performance measurements are optional validation.
