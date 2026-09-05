# Model internals of the broader pilot versus released weights

Development diagnostics run on 2026-09-06 under
[the instrumentation plan](internals-visualization-plan.md). No training was
run or repeated, every checkpoint hash was verified before and after inspection,
and nothing here is a pass/fail gate. Values are descriptive measurements on the
pilot's 512 fixed held-out validation windows (2,048 frames) in float32, eval
mode, with saved BatchNorm buffers. Raw ledgers: `runs/diagnostics/pusht_internals/<label>/`
(`internals.json`, `manifest.json`, `panels/*.png`); the dashboard indexes them.

## Headline comparison

| Measurement | pilot step 0 | step 250 | step 500 | step 1000 | released weights |
|---|---:|---:|---:|---:|---:|
| Effective rank of embedding covariance (of 192) | 17.6 | 9.8 | 12.5 | 14.1 | 66.2 |
| Participation ratio | 8.1 | 7.3 | 9.3 | 10.2 | 52.1 |
| Top eigenvalue share | 0.30 | 0.24 | 0.17 | 0.16 | 0.046 |
| Median per-dimension std | 0.0011 | 0.92 | 0.85 | 0.80 | 0.99 |
| Mean Shapiro–Wilk W (dims below 0.95) | 0.991 (0%) | 0.984 (0%) | 0.987 (0.5%) | 0.985 (2%) | 0.990 (0%) |
| Mean AdaLN gate, attention / MLP | 0 / 0 | 0.10 / 0.12 | 0.10 / 0.16 | 0.10 / 0.18 | 0.10 / 0.45 |
| Action embedding norm | 1.6 | 7.1 | 8.3 | 9.2 | 22.8 |
| Sensitivity to actions ÷ to state | 0 | 0.52 | 0.67 | 0.91 | 2.56 |
| Predictor attention entropy (0 sharp, 1 uniform) | 0.53 | 0.54 | 0.53 | 0.44 | 0.20 |
| Linear probe R², mean over 8 state targets (held out) | −0.25 | −0.07 | −0.05 | −0.06 | 0.66 |
| Rollout error ÷ copy-first-state error at horizon 8 | 29,758 | 11.2 | 2.6 | 0.65 | 0.10 |
| One-step error ÷ copy-previous-state error at horizon 8 | 62,025 | 24.4 | 3.2 | 0.88 | 0.040 |
| Gradient norm on one batch: encoder / predictor | 1.2 / 0.015 | 6,463 / 1.8 | 722 / 0.70 | 133 / 0.15 | 0.48 / 0.082 |

Per-target probe R² at step 1000: agent x 0.22, agent y −0.07, block x 0.19,
block y 0.25, block angle sin −0.38, cos −0.18, agent vx −0.22, vy −0.32.
Released weights: 0.95, 0.94, 0.97, 0.96, 0.78, 0.91, −0.17, −0.02. Velocities
are not linearly readable from a single frame in either model, as expected.

## Reading

- **The pilot representation is dimensionally collapsed and stays so.** Effective
  rank falls from 17.6 at random initialization to about 10 by step 250 and only
  recovers to 14 by step 1000, against 66 for the released weights. Per-dimension
  marginals are unit-scale and Gaussian (median std 0.8, W 0.985), so SIGReg is
  shaping each projection while the covariance stays concentrated in a dozen
  directions. The spectrum panel shows a cliff after roughly 12 components.
- **Physical state is not linearly readable from the pilot latent.** Held-out
  probe R² is at or below zero for every target, versus 0.78 to 0.97 for object
  positions and angle with the released weights, on the same windows and probe.
  This is the sharpest difference measured, and it is scale-free.
- **The pilot predictor uses actions less than state.** Action-to-state
  sensitivity rises from 0 to 0.91 over training but stays below 1; released
  weights sit at 2.6 with a 2.5× larger action embedding norm and 2.5× larger MLP
  gates. Attention gates are equal, so the difference is in how strongly the
  conditioning is applied, not whether it is wired.
- **Multi-step prediction barely beats copying.** At horizon 8 the pilot's
  autoregressive error is 0.65 of the copy-first-state error; the released model
  is at 0.10. The pilot's one-step error is 0.88 of the copy-previous-state
  error, so even teacher-forced prediction is close to "predict no change".
- **The objective is dominated by the encoder.** At every trained checkpoint the
  per-batch gradient norm of the encoder exceeds the predictor's by two to three
  orders of magnitude (eval-mode measurement, no update). The predictor's
  parameter norm moves from 118.9 to 119.6 over 1,000 updates.
- **Encoder attention is sensible in both models.** CLS-to-patch attention lands
  on the T-block and agent; the released patch PCA shows smooth spatial structure
  where the pilot's is flat outside the objects.

These observations are consistent with the earlier control failure: a low-rank
latent that does not encode object pose cannot support a useful planning cost.
They do not establish a cause (data scale, schedule, bf16 slicing, or recipe);
the source-scale reproduction remains the prepared next experiment.

## Method notes and limitations

- Ridge probe (λ = 1e-3) fitted on 1,024 training windows' first-frame latents,
  evaluated on the 512 held-out windows; held-out episodes are distinct initial
  configurations, so this measures transfer, not fit.
- Sensitivity is a central finite difference along 8 random unit directions in
  normalized action or latent space; gate magnitudes are batch means of |gate|.
- Attention maps use the pinned ViT with eager attention temporarily enabled;
  the state digest and checkpoint hash are unchanged afterwards.
- Gradient norms are eval-mode measurements on 8 windows, so BatchNorm uses
  saved buffers and no dropout mask; they differ from a training step.
- Latent-scale quantities (embedding std, absolute errors, norms) are not
  comparable across separately trained encoders; ratios and ranks are.
- Pilot inspections ran with the inspection script at commit `1fc6987` before it
  was committed in `eea734f`; the released-weights inspection ran at `eea734f`.
  Script content is identical.

## Dashboard changes delivered with this work

The canonical reader applies only filters that target every dataset, so the
earlier per-section selectors were silently ignored and all charts mixed every
run on a categorical axis. Charts now show one named selection fixed at build
time (`python -m viewer.dashboard --focus <training run>`), with per-scalar
training panels, a log10 validation-ratio chart, checkpoint-internals charts,
error-versus-horizon ratios, an all-checkpoint spectrum, embedded PNG panels
and an exact internals table. The static no-JavaScript fallback has a 750 KB
SVG budget and renders charts beyond the first eleven as data tables; the
interactive reader renders all of them. An opt-in `introspect: true` training
key records the scalar subset at every validation step for future runs.
