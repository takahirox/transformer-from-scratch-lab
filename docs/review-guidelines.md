# Review Guidelines

Review the PR against the source Issue using the [development flow](development-flow.md), including its English collaboration-language convention. These steps apply to AI agents as well as human reviewers.

## Review against the Issue

Read the source Issue, its mandatory acceptance criteria, and relevant clarifications before reviewing the PR description and full diff. The central question is:

> **Does the PR completely solve the source Issue, without unnecessary scope or complexity?**

## Check for missing work

Verify that every required behavior and acceptance criterion the PR claims to resolve is implemented and appropriately validated. Follow relevant links and check that documentation matches the actual repository structure and tooling.

Do not approve a PR as closing an Issue while required work remains. If the implementation is intentionally partial, its description must say so and the Issue must remain open.

## Check for unnecessary work

Reject additions without a demonstrated need tied to the Issue, including:

- Unnecessary abstractions or speculative extensibility.
- Unrelated refactoring.
- New frameworks or subsystems.
- Extra policies or configuration.

Prefer the smallest coherent design that fully satisfies the Issue. Added complexity is not automatically beneficial. Preserve justified existing conventions, and do not use simplicity to excuse omitted requirements or necessary validation. In this educational repository, preserve small, inspectable experiments and useful learning material.

## Check the result and evidence

Verify correctness, consistency with the existing source and documentation, and validation appropriate to the changed files. Read actual results and confirm failures, skipped checks, and unperformed checks are disclosed accurately. Do not infer successful validation from a proposed command.

Review applicable interface changes against the [UI/UX principles](../README.md#uiux).

Apply the [test and CI performance principles](development-flow.md#test-and-ci-performance-principles): consider the additional defects covered, overlap with existing checks, and execution/maintenance cost of new or expanded validation. Ask: **Does this change provide sufficient confidence with the lightest appropriate validation, without redundant execution?**

For templates/documentation-only changes, check template/Markdown syntax, working links, whitespace, and consistency. Follow the [available validation guidance](development-flow.md#choose-relevant-validation); do not require unrelated runtime suites, model training, hardware checks, or new tooling. For experiment changes, check the affected mathematics, tensor shapes, and recorded outputs as required by the Issue.

## Check pre-merge acceptance and post-merge verification

Require mandatory pre-merge acceptance to be achievable and verifiable before merge, with sufficient passing evidence. Keep optional validation separate. Checks possible only after merge, such as merge-triggered deployment or published-site verification, must not be pre-merge approval conditions.

Record genuinely necessary post-merge verification separately as pending until performed, or state that none is required. Do not add it routinely. Pre-merge evidence does not prove deployment or publication succeeded. This distinction does not waive implementation requirements, applicable pre-merge checks, or genuinely required post-merge work.

## Choose the review outcome

- **Agent-fixable findings:** request concrete fixes, identifying the affected file or behavior, the unmet requirement, and the expected result. Review the updated PR and validation again after revision.
- **Waiting for necessary human input:** record the specific question, why the answer is necessary, and what confirmation or decision is expected. Use the workflow's wait path and resume review when the answer arrives. Keep this state distinct from fixable code findings; do not repeatedly request code changes for missing human confirmation.
- **Ready to merge:** approve and merge when the implementation fully addresses the Issue it claims to resolve, no unjustified scope or complexity remains, all mandatory pre-merge checks pass with sufficient evidence, and no blocking findings or necessary unanswered human questions remain. Act within the task's authorization and repository permissions.

Pending checks possible only after merge do not prevent approval when those conditions hold. After merge, perform any recorded required verification and honestly report actual results and follow-up. Do not mark it complete until performed. Templates/documentation-only work normally requires no post-merge verification.
