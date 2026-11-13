---
name: Issue
about: Report a problem or propose a change
title: ""
labels: ""
---

<!-- Write the title and body in English. Remove prompts when filling this out. -->

## Problem

Describe the problem and why it matters.

## Expected outcome

Describe what should be true when the Issue is resolved.

## Context and reproduction

Add relevant examples, logs, related Issues, and reproduction steps with expected and actual results where applicable. For experiment changes, include relevant tensor shapes, inputs, and environment details.

For a defect discovered after merge, check for an existing equivalent Issue first. Include the affected revision and links to the original Issue and merged PR; track follow-up here without reopening the original solely for post-merge verification.

## Pre-merge acceptance

List concrete mandatory completion criteria that can be achieved and verified before merge.

- [ ] Describe a verifiable result.

Default to criteria an AI agent can execute and verify. Require human confirmation, physical-device testing, subjective evaluation, or external approval only when necessary for this Issue; explain why and what result is expected. See the [development guidance](https://github.com/takahirox/transformer-from-scratch-lab/blob/main/docs/development-flow.md#1-start-with-an-issue).

Checks possible only after merge must not be prerequisites for pre-merge approval. Keep all implementation requirements and applicable pre-merge checks.

## Optional validation

List useful additional checks separately from mandatory acceptance, or state that none are needed.

## Required post-merge verification

Record genuinely necessary checks possible only after merge separately, or state that none are required. Do not add them routinely. Close the fully implemented Issue after merge once mandatory pre-merge acceptance is satisfied, then perform these checks. Report unperformed or environment-blocked checks as pending with reasons; closure does not claim they passed. Follow the [merge and follow-up lifecycle](https://github.com/takahirox/transformer-from-scratch-lab/blob/main/docs/development-flow.md#5-merge-and-report-any-required-follow-up) for evidence recording and linked follow-up Issues.
