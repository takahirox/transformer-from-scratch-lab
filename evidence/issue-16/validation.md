# Issue #16 pre-merge validation

Validated locally on 2026-10-20 for [Issue #16](https://github.com/takahirox/transformer-from-scratch-lab/issues/16). Implementation and pre-merge checks are complete. **Actual GitHub Actions execution, Pages publication and public HTTPS validation remain pending until merge to `main`.** No push, PR, merge, Issue comment, repository setting change or publication was performed.

Environment: macOS 26.6.2 arm64, Apple M4 Max, Python 3.13.13, PyTorch 2.14.1, NumPy 2.4.2, ONNX 1.20.1, native ONNX Runtime 1.24.2, Node 24.12.0, npm 11.6.2, ONNX Runtime Web 1.22.0, Playwright 1.58.2, installed Google Chrome 155.0.8059.85. [Environment/configuration](environment.json) records CPU training and source hashes. The workflow targets Ubuntu 24.04 and the CPU PyTorch wheel; that remote environment has not been executed here. The Linux CPython 3.13 x86_64 CPU wheel was confirmed available with pip's cross-platform dry run.

## Commands and results

From the repository root, after `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements-export.txt` and `npm --prefix web ci`:

```sh
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
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/pages.yml
.venv/bin/python -m pip check
git diff --check
```

- **9 Python tests passed**, none skipped. Original attention script passed: shape `(3, 3)`, row sums `[1, 1, 1]`, output shape `(3, 2)` and expected hand-computed values. No Python implementation changes were needed.
- **4 JavaScript tests passed**, including rejection of missing WASM, changed model hashes, oversized artifacts, missing references, root-relative/missing source asset URLs, exclusion of unnecessary model/vendor files, and removal of stale artifacts on failed builds.
- Seeded training completed using the actual classifier: final held-out accuracy **1.0**, loss **0.0089382293** at training length 32. This synthetic evaluation is not a general accuracy guarantee. All four ONNX exports passed graph checking and all **12** native ONNX/PyTorch reference cases (`atol=rtol=1e-5`, identical predicted classes). Maximum absolute errors: logits **1.5497208e-6**, attention **5.3644180e-7**. Expected legacy-export deprecation and fixed-shape tracing warnings were emitted, as documented for the existing exporter.
- The assembled artifact contains **16 files**, **24,815,049 bytes (23.67 MiB)** against the **52,428,800-byte (50 MiB)** build limit. Four ONNX models total **191,064 bytes**; the site total includes all runtime JS/WASM, references and notices. [Build summary](build-summary.json) records every packaged file's size/hash, the manifest, training history, native comparison maxima and repeatability results. No checkpoints, environments, node_modules or training reports are in the site.
- **4 Playwright checks passed**, none skipped, against the packaged artifact under **`/transformer-from-scratch-lab/`**: visible unsupported-WebGPU state; actual unsupported-operator rejection with CPU fallback disabled; real WebGPU predictions/logits/attention at every exported length with input/head controls; and HTTP fetch/path/MIME/hash checks for all packaged assets. [Test outcomes](browser-tests.log) retain the actual final run (5.7 seconds); repetitive local HTTP access lines were omitted. The initial browser favicon request received an incidental 404; it is not an application asset or inference dependency.
- The real browser used a **non-fallback Apple `metal-3` adapter**, with no software-GPU or unsafe-WebGPU flags. All **12** fixture cases passed (`atol=1e-4`, `rtol=1e-3`, exact predictions), with maximum absolute errors **2.0861626e-6** for logits and **6.5565109e-7** for attention. [Browser report](browser-webgpu.json) records adapter, user agent, case inputs, errors, timings and local-only request origins. The browser's compatibility user-agent string does not identify the actual host CPU; use the environment record for that.
- A second full run with the same CLI arguments, using `results/pages-repeat/transformer.pt` and exporting to `results/pages-repeat/model`, produced **identical training results, checkpoint hashes, complete manifests, ONNX hashes and reference JSON**. Environment timestamps and benchmark timings were excluded from comparison. This demonstrates same-machine repeatability; cross-platform bit identity is unverified.
- Actionlint **1.7.12** passed with no findings; pip reported no broken requirements. Reviewed the full diff against Issue #16, including main-only deployment guards, PR read permissions, deployment-only Pages/OIDC writes, environment/concurrency, allowlist packaging and documentation. Whitespace checks passed including staged new files. Dependencies were installed with the existing pinned manifests; `npm ci` reported zero vulnerabilities.

No GPU tests were skipped locally. Hosted CI may lack a WebGPU adapter: the existing tests explicitly skip the two GPU-dependent cases, retain a capability probe and JUnit skip reasons, and still require static/unsupported-state checks. Such a run does not verify WebGPU execution. Other physical GPUs/browsers, caching/performance and cross-platform repeatability remain optional and unverified.

## Required post-merge verification: pending

After merge, follow [the deployment checklist](../../web/README.md#required-post-merge-verification-pending-until-publication). A repository administrator must select **Settings → Pages → Build and deployment → Source: GitHub Actions** and allow `main` in the `github-pages` environment. This work did not inspect or change current admin settings.

Wait for the actual main publication run and record its URL, completion/logs and deployed URL. Then request **https://takahirox.github.io/transformer-from-scratch-lab/** and verify the page, model/manifest/reference and runtime assets under that subpath, plus initialization/unsupported status and real-WebGPU inference where available. These are required post-merge checks, not a pre-merge gate or a claimed successful deployment. If publication fails, assets are broken or an admin change blocks it, create a new `task`-labeled Issue linking #16, the merged PR and deployment run, with symptoms, reproduction, expected behavior, the exact admin action when applicable and actionable pre-merge criteria; fix it in a separate PR.
