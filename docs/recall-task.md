# Historical recall in the editable recipe

The implemented path reads a complete session of canonical text observations,
then predicts where a queried entity was last observed. Four location classes and
`not_observed_in_session` are factual outputs. Abstention is a separate decision,
chosen when answering has no lower estimated cost. This is a controlled memory
task, not visual mapping or general instruction understanding.

## Run it

For the bounded current/recent learning diagnostic, use:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --dataset recall --recall-mode current-recent --output runs/my_recall_diagnostic
```

This mode defaults to four-event sessions, 40 fixed training episodes, 100 fresh
development episodes, seed 17, batch size 4 and 256 updates with a cumulative
900-second active-time cap. It logs training factual scores every 16 updates and
evaluates the final checkpoint once on development. Calibration and final-test
splits are never loaded. The time cap is checked before each update/evaluation
episode and survives pause/resume; an in-flight operation can finish. Read the
[declared pilot and routing gates](recall-learning-plan.md) before changing defaults.

Use `--check` for a forward/backward check or `--stop-after N` and then
`--resume runs/my_recall_diagnostic` for an intentional pause. A stopped run with an
exhausted time budget stays stopped. Its complete split manifest, raw metrics,
final logits, per-class/cohort results and routing gates live beside `last.pt` and
the self-contained `report.html`. Completed report rebuilds reuse cached logits.

For the original longer historical task:

Use the same recipe and run/report machinery as the multimodal tasks:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --dataset recall --history 16 --memory-recent 2 --memory-block 2 \
  --memory-blocks 2 --recall-truncate 8 --steps 8 --batch-size 1 \
  --output runs/my_recall_check
```

Choose a new output directory. `--stop-after 1` pauses after one update; resume
with `--resume runs/my_recall_check`. Resume requires compatible source, settings,
data and environment. A completed resume rebuilds the report from cached final
predictions without refitting or rerunning held-out inference.

`--dataset recall --check --batch-size 1` performs one forward/backward check using
the default 256-record sessions and default memory capacities. It prints diagnostic
JSON and does not train. The 16-record command above uses deliberately smaller
memory so source records pass through compression and consolidation quickly.

Each split defaults to 15 independent episodes. Counts must be positive multiples
of 15: five balanced factual labels, with seen examples also balanced across recent,
compressed and consolidated reference groups. `--train-windows`,
`--validation-windows`, `--calibration-windows` and `--test-windows` set these counts.
The existing `validation` name means development/model selection in this task.

## Inputs, outputs and evidence

`RecallEpisodes` in the recipe owns canonical text such as
`saw entity=e07 at location=l2`. Each episode has a fresh session ID, independent
entity/location bindings and a randomized absolute time offset. The generator
privately conditions a history on its balanced label/retention stratum; only after
the entire history has been replayed does inference receive the query. Group, label,
generator seed and hidden trajectory are absent from learned inputs. An earlier
appearance has an independently sampled location, with no fixed mapping to the answer.

The policy replays ordered observations through normal event transactions. It has
no evaluator lookup or seen-entity table. At the fixed session cutoff,
`recall_logits(model, state, query)` encodes the entity question and objective,
performs two thinking/memory-read rounds, and applies a small five-class head to
the working tokens. Opaque IDs and arbitrary task text do not supply label shortcuts.
This first exact query adapter does not yet interpret arbitrary user instructions.

`RecallQuery` and `RecallDecision` beside existing task records round-trip through
plain dictionaries. `select_recall` uses the explicit correct/wrong/abstain cost
rule. With default costs 0/1/0.25, it answers only above probability 0.75; an exact
tie abstains. Invalid probabilities are logged as invalid abstentions.

`verify_recall` checks the actual emitted decision against an independently supplied
complete delivered log and inclusive cutoff. It returns verified, contradicted or
unknown; abstention is unknown, not successful recall. It rejects missing prefixes,
wrong sessions and mismatched tasks/cutoffs. The authoritative log belongs to the
harness, not the online agent. Hidden changes in current location cannot change a
historical target. Generated answers never become observation records.

## Training and finalization

Every query receives factual cross-entropy supervision, including those on which
the decision would abstain. Training samples independent episodes uniformly.
Final-segment text reconstruction, source reconstruction and split categorical KL
anchor state learning. No fake image, audio or physical action targets are added.

The complete prefix executes with the same weights and state under `no_grad`.
Only the last `--recall-truncate` records retain autograd graphs (zero means the
whole history). Forward memory is never reset at that boundary. Thus the last
segment can train current readers and consolidation, but delayed answer loss does
not directly train detached earlier encodings. Full-horizon credit assignment is
still an open design decision; this initial method makes its limitation explicit.

Development factual NLL is measured at initialization and every update. The lowest
value selects the model, with ties keeping the earlier checkpoint. In recall runs,
`LearningState.target` holds those selected weights rather than an EMA teacher.
Selection state is saved with the optimizer and sampler, so pausing cannot introduce
additional candidate checkpoints or change the selected result.

After all updates, the selected model produces calibration logits. One positive
temperature is fitted by a deterministic bounded search, over [0.05, 20], including
T=1 as a candidate. Calibration failure keeps T=1 with a visible failure status.
The selected model then produces test logits once. Raw and adjusted results and
the three predeclared abstention-cost views reuse those logits. Neither calibration
nor test results select model weights. The temperature, selected step and selected
weights live in the safe checkpoint; split and model identities bind the cached
predictions. No calibration or held-out inference happens on a paused run.

## Run artifacts and interpretation

`last.pt`, `run.json`, `metrics.jsonl` and source snapshots use the existing pipeline.
Additional files are `recall_predictions.pt` (safe tensor/dictionary cache),
`recall_results.json` (all metrics and auditable examples), and
`recall_performance.json` (descriptive wall-clock inference time and operation
counts). A completed run has its own offline `report.html` with confidence charts,
population tables, baselines and fixed representative examples. Result completion
is written independently of report success; report failures remain visible.

Report factual NLL/accuracy, task loss, answer coverage and error among answers
together. The latter is null with no answers. Seen-old groups matter: perfectly
recognizing only unobserved entities and abstaining on every seen entity already
scores 0.20 in this population without remembering a location. The report includes
that oracle reference as well as all-abstain (0.25), fixed-class answer and full-log
oracle references. The oracles are not online agent components.

Retention groups describe the original source position under the declared memory
schedule, not which store caused a correct answer. Current recurrent state may
carry old information. A trained memory-restricted diagnostic and matched trained
controls are required for a hierarchy-benefit claim. Temperature scaling and tiny
development samples supply no reliability guarantee. See the original
[task design](recall-task-design.md) and [implementation record](recall-implementation-plan.md).
