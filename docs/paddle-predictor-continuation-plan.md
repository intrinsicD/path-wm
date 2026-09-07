# Bounded predictor continuation — 2026-09-07

## Reference and hypothesis

The fixed reference remains immutable in `runs/paddle/baseline`. Perception and
memory completed10000 updates. P1 also completed10000 and failed the original
gate: latent0.273366 versus copy1.077386, H x/y/paddle3.70127/3.59783/1.79505
versus copy3.83335/2.38499/2.52450. Only ball-y fails. Its objective and y error
were still improving late in the budget. This motivates a bounded test of further
optimization, without claiming that additional updates must resolve the failure.

The user requested iteration to a working world model. This follow-up tests
optimization separately from the optional training-start change currently being
discussed. It preserves E/U/P/D/H/R, losses, data, statistics, validation cases,
checkpoint selection and the original gate. It is additional compute beyond the
failed reference budget and must never be labelled successful reference training.

## Intervention and budget

Fork the selected immutable P1 checkpoint into
`runs/paddle/continuation_v1/predictor_1`. Preserve weights, optimizer, RNG/sampler,
best-validation criterion, selected metrics and cumulative example/time counters.
Change only the P1 maximum update from10000 to20000, allowing at most10000
additional updates of batch64 with the same AdamW3e-4 and eight-check early stop.
Configuration: `configs/paddle/predictor_continuation.yaml`.

The fork records parent SHA256, path, model fingerprint, original configuration,
source update/examples/time, and the additional budget. Source files are never
rewritten. A new directory is published atomically only after validation. Resuming
the child must retain this provenance and the cumulative versus added work.

At each250-update validation, compare the same1024 windows with copy-S. Select
the lowest objective, exactly as before. Proceed only if selected latent loss
and every H x/y/paddle MAE are strictly below matched copy. If it passes, initialize
the same P at K5 with a fresh optimizer and the unchanged10000-update budget;
then run the prescribed500 ordinary starts and100 pairs with all five policies.
If it fails, keep the failure and diagnose before proposing another intervention.
No test results select continuation parameters or checkpoints.

## Essential verification and minimal slice

Before implementation, test rejection of non-P1 parents, changed non-budget
settings, invalid budgets, existing targets and parent overwrites. Check exact
weights/optimizer/RNG/statistics, immutable parent bytes, valid selected-snapshot
recovery, preserved counters and provenance through child resumption.
Use a tiny CPU fixture to verify that the next resumed update agrees with the
corresponding uninterrupted update. Commit the plan/tests after their informative
red result. Then implement the fork and provenance plumbing, run the focused
checks, and commit before the GPU continuation. Each completed stage refreshes
and verifies canonical HTML; raw ledgers and diagnostics remain authoritative.

## Measured completion and evaluation implementation amendment

P1 continuation passed the original gate at20000 cumulative updates after10000
additional updates: H2.42536/2.22583/1.65811 versus copy3.83335/2.38499/2.52450,
latent0.190358 versus1.077386. K5 then completed10000 updates, selected10000,
in143.67s. Final-horizon H errors4.35023/4.03583/4.74094 improve over copy
16.0558/11.3849/5.59907 but exceed the two-pixel engineering target. Full control
evaluation is still required and has not begun.

Before launching full control, a measured implementation optimization removes
unused RGB rendering from privileged candidate simulations. Twenty validation
states over three repeats measure baseline21.43ms/decision; an isolated prototype
using identical physics and exact scalar prefix accumulation measures2.87ms.
Rendering accounts for about71% of baseline time. NumPy batch score reduction
changed8/4860 candidate scores by1–2 ULP and was rejected. The accepted path keeps
the original per-step Python arithmetic and fixed sequence ordering.

Essential tests must establish exact state/events/counters and all candidate
scores/actions, caller/RNG preservation, terminal/truncated behavior and absence
of unused rendering. Extract the existing integrator into `advance(action)` and
keep `step(action)` as its RGB-returning wrapper. This does not alter data,
physics, action/horizon protocol, learned planning, models or target thresholds.
Record the privileged implementation in evaluation hardware/latency metadata.
The full evaluation starts only after parity tests pass and this slice is committed.
