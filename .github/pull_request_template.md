<!-- Write the title and body in English. Remove prompts when filling this out. -->

## Summary

Describe the source problem and the focused changes that address it.

## Outcome

Describe the resulting behavior, including a before/after example when useful. Explain how the change satisfies the source Issue; state any intentionally partial scope and remaining work.

## Validation

List the checks actually performed and their results, including failures. Identify skipped or unperformed checks and explain why. Choose checks appropriate to the changed files and existing repository tooling. Separate optional checks from mandatory pre-merge acceptance.

## Required post-merge verification

List genuinely necessary checks possible only after merge as pending, or state that none are required. Do not claim deployment or publication succeeded based on pre-merge validation. After successful merge and Issue closure, record actual results, including unperformed or environment-blocked checks and reasons, in this PR, the closed Issue, or a linked validation report. Follow the [merge and follow-up lifecycle](https://github.com/takahirox/transformer-from-scratch-lab/blob/main/docs/development-flow.md#5-merge-and-report-any-required-follow-up) for discovered defects.

## Related issues

Link the source Issue and other relevant Issues. Use and preserve `Closes #<number>` when the implementation fully addresses that Issue and mandatory pre-merge acceptance is satisfied, even if post-merge-only verification is pending. For intentional partial work, use a non-closing link and leave the parent Issue open.
