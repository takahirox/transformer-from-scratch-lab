# Development Flow

This is the default Issue → implementation → PR → review → revision as needed → merge flow for Transformer From Scratch Lab, including AI-assisted work. Use the [review guidelines](review-guidelines.md) for review and merge decisions and the [test and CI performance principles](#test-and-ci-performance-principles) when choosing validation.

## Collaboration language

Use **English** for newly authored templates, development/review/merge documentation, AI instructions, Issue and PR titles and bodies, review comments, and commit messages. This collaboration-language convention does not require translating historical content or changing product localization.

## 1. Start with an Issue

Use the [Issue template](../.github/ISSUE_TEMPLATE/issue.md) to capture the problem, expected outcome, relevant context, reproduction steps where applicable, and concrete completion criteria. An assignee or owner is not required.

Treat the source Issue as the scope of the work. Inspect the current default branch (`main`) and reuse equivalent existing files and conventions. Clarify genuinely blocking ambiguity before implementing the affected work; do not invent requirements.

Default to completion criteria an AI agent can execute and verify. Require human confirmation, physical-device testing, subjective evaluation, or external approval only when necessary for the actual Issue. Explain why it is necessary and what result is expected.

Keep three categories separate:

- **Pre-merge acceptance:** mandatory implementation and checks achievable and verifiable before merge.
- **Optional validation:** useful additional checks that are not required for completion.
- **Required post-merge verification:** genuinely necessary checks possible only after merge, or explicitly none. Do not add these routinely or make them pre-merge approval conditions.

For example, merge-triggered deployment or published-site verification belongs after merge if an Issue actually involves it. This distinction does not waive implementation requirements or applicable pre-merge checks.

## 2. Implement the smallest complete change

Prefer the smallest coherent design that fully satisfies the Issue. Preserve justified existing conventions and useful documentation. Avoid unnecessary abstractions, speculative extensibility, unrelated refactoring, frameworks/subsystems, or extra policies/configuration without a demonstrated need tied to the Issue. Simplicity must not excuse missing requirements or necessary validation.

This repository is an educational lab built around small, inspectable experiments: [the step-by-step self-attention experiment](../01_attention_basics/self_attention_step_by_step.py), the [integrated Python/PyTorch baseline](../baseline/README.md), and the [browser-local WebGPU demo](../web/README.md). The [README](../README.md) describes the learning goals and experimental method; its roadmap extends beyond the current implementations.

### Choose relevant validation

Python dependencies are pinned in [requirements.txt](../requirements.txt) and [requirements-export.txt](../requirements-export.txt); JavaScript commands are in [web/package.json](../web/package.json). The [Pages workflow](../.github/workflows/pages.yml) runs Python/JS checks, seeded CPU training, ONNX export validation, packaging and Playwright checks for relevant PRs to `main`. Relevant pushes to `main` also build and publish; manual dispatch validates the selected branch and publishes only on `main`. Root README and `docs/` edits alone do not trigger it. No Markdown linter is configured. Use existing tooling and relevant checks:

- For templates/documentation-only changes, inspect Markdown structure and template front matter, verify links and consistency, and run `git diff --check` for whitespace. Check newly added files as well as tracked edits. Use an already available Markdown parser or linter if useful. No model training, new runtime tests, hardware/device checks, or deployed-site verification is required for this scope.
- For changes to the existing experiment, when Python and PyTorch are available, run `python3 01_attention_basics/self_attention_step_by_step.py` from the repository root and inspect the affected intermediate values, tensor shapes, and attention row sums against the expected mathematics. Record the actual outputs and any relevant environment details. The current script uses CPU tensors; GPU checks or training are required only when the source Issue makes them necessary.
- For core/model/checkpoint/export changes, use `.venv/bin/python -m unittest discover -s tests -v` after the [documented setup](../baseline/README.md#setup-and-commands). Export checks need `requirements-export.txt`; validate affected exported graphs and numerical references as described in the [web checks](../web/README.md#reproducible-checks).
- For UI/browser-runtime changes, use `npm --prefix web test` for pure JavaScript logic, plus `npm --prefix web run build` and `npm --prefix web run test:browser` for affected browser behavior after generating the documented model assets. Python/JS success does not establish real WebGPU execution; report capability skips explicitly.
- For CI/build/publication changes, validate workflow syntax and the full applicable [Pages build/check sequence](../web/README.md#github-pages-build-and-deployment), including `build:pages` and `test:pages` under the repository subpath. Keep applicable PR checks mandatory; a successful build does not prove publication. Record genuinely required [post-merge publication checks](../web/README.md#required-post-merge-verification-pending-until-publication) separately. Additional browsers/GPUs and longer benchmark experiments are optional unless the Issue requires them.

Report failures, skipped checks, and unperformed checks accurately, with reasons. Do not install a training framework, download data, or require GPUs merely to follow this documentation.

### Test and CI performance principles

**Detect the required failures at the lightest appropriate layer, without redundant execution.** Preserve required correctness, regressions and meaningful integration coverage when redistributing checks.

1. **Prefer the lightest meaningful layer.** Use Python unit/component checks for tensor mathematics, shapes and branches, and Node tests for input parsing, comparisons and packaging logic. Reserve trained-model/ONNX and Chrome/WebGPU integration for checkpoint, numerical, device and browser boundaries that require them. Test a concrete defect rather than mirroring implementation details.
2. **Choose combinations deliberately.** Select representative model architectures, tensor shapes, masks, sequence lengths and browser/hosting paths for distinct processing paths; exhaust combinations when their interactions require it. Preserve the existing every-exported-length numerical references and meaningful browser regressions; avoid automatically multiplying every architecture, device, length and browser.
3. **Match validation to the change.** Use the [validation guidance above](#choose-relevant-validation) to separate documentation, UI, core/runtime and CI/build checks. Local focused checks give fast feedback; applicable PR build/checks provide regression confidence; main publication and genuinely post-merge-only checks have separate responsibilities. Distinguish mandatory pre-merge, optional and required post-merge work in the Issue and evidence.
4. **Reuse work and avoid duplicate execution.** Reuse compatible builds, trained checkpoints, exports and downloads with cache keys tied to inputs/tool versions and isolated outputs. The workflow already caches npm downloads and deploys its tested build artifact. Avoid equivalent runs for the same revision through overlapping push/PR triggers; the current main push also serves publication. Cancel superseded validation only when safe; current concurrency does not cancel runs and preserves production ordering. Measure resource use before increasing parallelism. Changes to caching, triggers, cancellation or workers need separately scoped implementation work.
5. **Review added confidence and cost together.** Identify the additional defect a new or expanded test catches, whether existing coverage already detects it, and its execution/maintenance cost. Watch repeated training/export, browser/session setup, large fixtures and Cartesian-product growth. Retain meaningful assertions and required coverage while removing redundant work.
6. **Maintain an execution-time budget.** A normal-CI baseline and target are not yet established. [Recorded local validation](../evidence/issue-16/validation.md) includes individual check timings, not a full CI budget; model inference benchmarks are a different measurement. Establish a repository-appropriate baseline/target during actual test/CI performance work, then monitor growth and investigate overruns. Track elapsed feedback time, total runner/computation time and queue time separately; revisit validation structure and duplication while retaining required confidence. Workflow timeouts are limits, not measured targets. This documentation adds no benchmark or merge gate.

## 3. Create a Pull Request for the Issue

Target `main` unless the task selects another base branch. Use the [PR template](../.github/pull_request_template.md) to explain the problem, resulting behavior, focused changes, related Issues, and validation actually performed. Use concise English commit messages describing what changed and why, and reference the Issue.

Claim to close an Issue only when its implementation requirements are fully addressed. State intentional partial scope clearly and leave the Issue open. Report completed pre-merge checks separately from required post-merge verification, which remains pending until performed. Pending checks possible only after merge do not make an otherwise complete implementation partial.

## 4. Review and revise as needed

Every PR should be reviewed before merge. The central question is: **Does the PR completely solve the source Issue, without unnecessary scope or complexity?** Check both missing requirements and unjustified additions using the [review guidelines](review-guidelines.md).

Request fixes for agent-fixable findings, then review the updated PR again. If genuinely necessary human input is missing, record the specific question and expected answer and wait through the workflow's wait path. Distinguish waiting for human input from fixable findings; do not repeatedly request code fixes for an unanswered human question.

## 5. Merge and report any required follow-up

Approve and merge when the implementation is complete, mandatory pre-merge checks are sufficient and pass, and no blocking review findings remain, within the task's authorization and repository permissions.

Perform recorded required post-merge checks when applicable. Keep them pending until performed, then report actual results, failures, and necessary follow-up. Pre-merge validation does not prove deployment or publication succeeded. Templates/documentation-only work has no required post-merge verification unless the source Issue establishes a specific need.
