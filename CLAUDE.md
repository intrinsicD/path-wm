# Working instructions

## Current objective
Build a fresh modular implementation of the published LeWM baseline. Demonstrate
learning and control on PushT and support multiple explicit dataset protocols.
Research extensions remain deferred until the reference baseline works.
The user explicitly requested removal of the previous code and results; only
ideas in docs/ideas.md and downloaded source datasets carry forward.

## Workflow
Read this file first. State the active implementation/validation task. Use small
end-to-end steps: define interfaces and configuration, write essential behavioural
tests, implement, then measure learning and planning. Commit completed steps.
Do not build speculative frameworks, registries, or deferred research modules.
Use plain PyTorch modules, functions and explicit configuration. Components must
be replaceable without changing the training loop or dataset adapters.

Reuse the pinned authors' baseline components before inventing replacements.
Copied/adapted code must carry source URL, commit, file and upstream license.
Document deviations from the reference recipe and distinguish smoke checks,
subset training, reproduction training and benchmark evaluation. Never present
falling loss alone as evidence of a useful world model. Never hide failed gates.

Keep source data under data/ and all new checkpoints/logs under runs/. Never mix
source data, training episodes and held-out evaluation episodes. Record data and
code revisions, seed, configuration, normalization and sample/step counts.
No external experiment tracking or uploads unless requested.

Tests cover silent scientific failures: episode alignment, causal action timing,
reference computation/gradients, normalization, rollout and checkpoint integrity.
Keep ordinary tests fast and CPU-based; substantive GPU runs are validation, not
unit tests. Do not remove a failing test to conceal an implementation error.

## Status
Clean reset in progress. Published LeWM source and dataset revisions are pinned;
new baseline learning and control have not yet been established.
