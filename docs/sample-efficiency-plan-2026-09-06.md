# Sample-efficiency investigation — 2026-09-06

User request: research with Claude whether substantially fewer examples can suffice; inspect gradients, batch formation and sizes; develop a grounded theory and a plausible solution.

Preserve all training checkpoints and the source reproduction recipe. This investigation makes no optimizer updates. Use discarded model instances and fixed training windows for gradient probes. Run diagnostics through `run.py`, refresh the canonical dashboard, and retain raw results and hashes.

## Questions and predeclared measurements

1. Count unique episodes, windows, encoded frame identities and supervised transition identities; reconstruct the exact shuffled sampler. Measure overlap within batches separately from repetition across an epoch. Inspect trajectory/configuration grouping metadata where available.
2. Summarize existing logged training losses, pre-clip gradient norms, learning rates and timings. These are sampled logs, not every update.
3. At the current PushT (step 13933) and TwoRoom (4074) checkpoints, measure actual training-mode gradients on four independent, frozen batches, with nested batch sizes 32, 64 and 128. Separate prediction and weighted SIGReg gradients, report module norms and cosines, total-gradient consistency across batches and same-batch stochastic repeats. Preserve BatchNorm buffers between probes and fix random seeds. BatchNorm and SIGReg couple examples: these probes cannot identify a universal critical batch size or treat microbatch accumulation as equivalent.
4. On a fixed batch of 128, decompose the prediction gradient through the target and input branches. This tests moving-target cancellation as a hypothesis, without claiming that stop-gradient would preserve collapse avoidance.
5. Check a discarded AdamW update analytically using saved moments (including clipping) if needed to interpret large raw gradients. Raw norm dominance alone does not establish optimizer-step dominance.

## Collaboration, literature and budget

Claude contributes an independent derivation, competing hypotheses and generic diagnostic implementation through the registered MCP tool. Use a neutral directory, no repository access, and only public research/generic mathematical inputs. Apply its work locally and verify citations independently. At most two calls, each with a $2 reported API-equivalent cap; account balances are unavailable. No repository transfer is necessary.

Read a focused set of primary papers: LeWM/LeJEPA, gradient-noise and batch-scaling studies, and objective/normalization or data-selection papers motivated by measurements. Initial diagnostic budget: at most 30 GPU minutes, no new training or dataset downloads. Tests cover gradient algebra, batch identities and state preservation before implementation is committed.

## Decision and output

Write `docs/sample-efficiency-2026-09-06.md` with observed facts, literature support, causal uncertainty and a ranked, falsifiable solution. Distinguish unique-data efficiency, processed-example efficiency, update efficiency and wall time. Propose a minimal ablation with frozen disjoint evaluation configurations, matched unique-data and compute budgets, and unchanged control cases. Improvements remain hypotheses until a training learning-curve experiment tests them.
