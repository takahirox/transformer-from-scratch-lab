# Issue #18 publication verification

Completed on 2026-10-20 for [Issue #18](https://github.com/takahirox/transformer-from-scratch-lab/issues/18), following the publication checklist for [Issue #16](https://github.com/takahirox/transformer-from-scratch-lab/issues/16) and merged [PR #17](https://github.com/takahirox/transformer-from-scratch-lab/pull/17). **Pages publication and real-device WebGPU inference are verified.** This record supplements the historical [Issue #16 pre-merge evidence](../issue-16/validation.md).

## Administrator action and completed deployment

The initial read-only checks reproduced HTTP 404 from both the Pages API and the expected public index. The [original deployment job](https://github.com/takahirox/transformer-from-scratch-lab/actions/runs/39364610372/job/118438970564) failed at `actions/configure-pages@v5` with `Get Pages site failed` / `HttpError: Not Found`. The authenticated account, `takahirox`, had repository administrator permission. Actions was already enabled with `allowed_actions: all`; the `github-pages` environment had no protection rules or deployment branch restriction, so `main` was permitted.

The administrator action was performed through GitHub's [Pages REST API](https://docs.github.com/en/rest/pages/pages#create-a-github-pages-site), setting the source to GitHub Actions:

```sh
gh api --method POST repos/takahirox/transformer-from-scratch-lab/pages -f build_type=workflow
gh api repos/takahirox/transformer-from-scratch-lab/pages
gh run rerun 39364610372 --failed --repo takahirox/transformer-from-scratch-lab
gh run watch 39364610372 --repo takahirox/transformer-from-scratch-lab --exit-status --interval 5
```

The creation succeeded. A GET confirmed `build_type: workflow`, `https_enforced: true` and the expected HTTPS URL before the rerun, at **03:34:51 UTC**. The API also reports `source: {branch: main, path: /}`; `build_type: workflow` selects the Actions publishing source. No branch-based publishing or generated-asset commit was configured. Actions policy and environment rules required no changes.

The [main publication run, attempt 2](https://github.com/takahirox/transformer-from-scratch-lab/actions/runs/39364610372/attempts/2) finished with **success** at **03:35:12 UTC**, for merge commit `e3a749abe656817994c6fc4b19174b93aa253bae`. Only the failed deployment job was rerun; it reused the original successful build's `github-pages` artifact. The [successful deployment job](https://github.com/takahirox/transformer-from-scratch-lab/actions/runs/39364610372/job/118439859244) passed configuration with `enablement: false`, reported deployment success and returned **https://takahirox.github.io/transformer-from-scratch-lab/** as its environment URL. This is a completed production deployment, separate from the original pre-merge acceptance.

[Deployment metadata](deployment.json) and [bounded deployment log excerpts](deployment-excerpts.log) retain the configuration, run, jobs, timestamps and results. The deployment emitted non-fatal `punycode` and action Node.js-version deprecation warnings; neither prevented publication. The Pages API's `status` field was still `null` when recorded; success is established by the completed deployment and live HTTP/browser checks below.

## Published HTTPS artifact

Downloaded the original artifact with:

```sh
gh run download 39364610372 --repo takahirox/transformer-from-scratch-lab \
  --name github-pages --dir results/issue-18/artifact
```

Performed HTTPS GETs of the index and every packaged asset under `/transformer-from-scratch-lab/`, comparing each response body byte-for-byte with the corresponding file inside `artifact.tar`. All **16 files / 24,814,990 bytes (23.67 MiB)** returned **HTTP 200** at their exact expected URLs. Verified HTML/CSS/JSON MIME types, JavaScript MIME types for `.js` and `.mjs`, and `application/wasm` for WASM. Every ONNX file matched both the packaged bytes and the manifest's size/SHA-256.

Coverage includes `index.html`, `style.css`, `app.js`, `core.js`, `runtime.js`, `model/manifest.json`, its `reference.json`, all four `recall-{2,16,32,64}.onnx` files, the three vendor runtime JS/WASM files, and both notice files. The reference JSON covers all four lengths with **12 fixtures**. [HTTP evidence](http.json) records each URL, status, MIME type, byte count, SHA-256 and comparison result. Downloaded artifacts remain under ignored `results/`; model/runtime bundles are not committed.

## Browser and actual WebGPU inference

Used the repository's pinned Playwright **1.58.2** after `npm --prefix web ci` (zero reported vulnerabilities), launching installed **Google Chrome 155.0.8059.85** with `{channel: 'chrome', headless: true}`. No software-GPU or unsafe-WebGPU flags were supplied. Host: macOS **26.6.2**, arm64, **Apple M4 Max**. The compatibility user-agent string does not identify the host CPU. The browser obtained a non-fallback **Apple `metal-3`** adapter on the published HTTPS origin.

- The published app initialized and displayed **WebGPU ready · CPU compute fallback disabled**.
- Clicked the actual **Compare with PyTorch fixtures** control: all **12 cases** across lengths **2, 16, 32 and 64** passed logits/attention tolerances (`atol=1e-4`, `rtol=1e-3`) and exact predicted-class agreement. Maximum absolute errors were **1.2218952e-6** for logits and **4.1723251e-7** for attention.
- Selected each exported length, entered cycling token IDs, ran inference and switched to the last attention head. Each run displayed **Completed locally with WebGPU**, a prediction, and the expected number of attention rows with the selected head label. Predictions at lengths other than the training length can be incorrect; fixture agreement verifies execution, not recall accuracy at every length.
- Observed application requests stayed on `https://takahirox.github.io`; no off-origin request was allowed. There were no uncaught page errors, failed requests or unsuccessful application HTTP responses. The served source matches the tested client-only app; inference used the WebGPU provider with CPU compute fallback disabled.
- In a separate browser context, removed `navigator.gpu` before loading the same public page. The app displayed the clear **WebGPU is unavailable** message, disabled inference and hid results. This is a simulated unsupported-device check, separate from the real GPU pass.

[Browser evidence](browser.json) retains the environment, capability probe, initialization/status messages, every fixture result/error/timing, inference-control outcomes, request URLs/origins and unsupported-state result.

The original hosted build passed its static checks but had **2 browser passes / 2 GPU-dependent skips** because no WebGPU adapter was available. It was reused rather than rerun. [Build log excerpts](build-excerpts.log) preserve that distinction. The successful physical-device checks above independently establish execution of the published artifact; they do not turn the hosted skips into GPU passes. Other physical GPUs/browsers and performance benchmarking remain unverified. Timing samples are observations, not a benchmark. No application or workflow change was needed.

## Completion and repository validation

All operational verification required by Issue #18 is complete. The one-time administrator setup remains documented in [web/README.md](../../web/README.md#github-pages-build-and-deployment); workflow permissions, non-publishing PR checks and client-only inference are unchanged. This commit records evidence only. No push, PR, merge, Issue comment or Issue closure was performed.

Checked evidence JSON parsing/consistency, local Markdown links, and whitespace including staged new files with `git diff --cached --check`. Model training and source unit suites were not rerun locally because no implementation changed; the original successful build and the live checks above provide the relevant validation.
