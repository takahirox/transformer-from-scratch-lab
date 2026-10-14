# Development Flow

This is the default Issue → implementation → PR → review → revision as needed → merge flow for Transformer From Scratch Lab, including AI-assisted work. Use the [review guidelines](review-guidelines.md) for review and merge decisions.

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

This repository is an educational lab built around small, inspectable experiments. Its current source is [the step-by-step self-attention experiment](../01_attention_basics/self_attention_step_by_step.py), which uses Python and PyTorch. The [README](../README.md) describes the learning goals and experimental method; its roadmap is not a list of existing implementations or tooling.

### Choose relevant validation

There is currently no dependency manifest, configured test runner, Markdown linter, or CI workflow. Use the tooling actually available and checks appropriate to the changed files; do not invent commands or add setup solely to document a process.

- For templates/documentation-only changes, inspect Markdown structure and template front matter, verify links and consistency, and run `git diff --check` for whitespace. Check newly added files as well as tracked edits. Use an already available Markdown parser or linter if useful. No model training, new runtime tests, hardware/device checks, or deployed-site verification is required for this scope.
- For changes to the existing experiment, when Python and PyTorch are available, run `python3 01_attention_basics/self_attention_step_by_step.py` from the repository root and inspect the affected intermediate values, tensor shapes, and attention row sums against the expected mathematics. Record the actual outputs and any relevant environment details. The current script uses CPU tensors; GPU checks or training are required only when the source Issue makes them necessary.

Report failures, skipped checks, and unperformed checks accurately, with reasons. Do not install a training framework, download data, or require GPUs merely to follow this documentation.

## 3. Create a Pull Request for the Issue

Target `main` unless the task selects another base branch. Use the [PR template](../.github/pull_request_template.md) to explain the problem, resulting behavior, focused changes, related Issues, and validation actually performed. Use concise English commit messages describing what changed and why, and reference the Issue.

Claim to close an Issue only when its implementation requirements are fully addressed. State intentional partial scope clearly and leave the Issue open. Report completed pre-merge checks separately from required post-merge verification, which remains pending until performed. Pending checks possible only after merge do not make an otherwise complete implementation partial.

## 4. Review and revise as needed

Every PR should be reviewed before merge. The central question is: **Does the PR completely solve the source Issue, without unnecessary scope or complexity?** Check both missing requirements and unjustified additions using the [review guidelines](review-guidelines.md).

Request fixes for agent-fixable findings, then review the updated PR again. If genuinely necessary human input is missing, record the specific question and expected answer and wait through the workflow's wait path. Distinguish waiting for human input from fixable findings; do not repeatedly request code fixes for an unanswered human question.

## 5. Merge and report any required follow-up

Approve and merge when the implementation is complete, mandatory pre-merge checks are sufficient and pass, and no blocking review findings remain, within the task's authorization and repository permissions.

Perform recorded required post-merge checks when applicable. Keep them pending until performed, then report actual results, failures, and necessary follow-up. Pre-merge validation does not prove deployment or publication succeeded. Templates/documentation-only work has no required post-merge verification unless the source Issue establishes a specific need.
