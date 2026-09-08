# Frozen-encoder decoder recovery — 8 September 2026

Decoder refitting restored generic reconstruction almost to its original warmup quality without changing the task-adapted encoder. This establishes recoverability at RGB64 under this protocol; it does not establish that adaptation preserved every feature or that the original failure was caused exclusively by the decoder.

| Encoder / decoder | COCO test MSE ↓ | PushT test MSE ↓ |
|---|---:|---:|
| Original warmup pair | 0.006072331 | 0.001100839 |
| Task-adapted pair | 0.272195453 | 0.000187608 |
| Frozen adapted encoder, refitted existing decoder | 0.006104667 | 0.001635113 |
| Frozen adapted encoder, fresh decoder | 0.006389832 | 0.028031899 |
| Frozen warmup encoder, matched fresh decoder | 0.005058220 | 0.000962422 |

The existing-decoder refit reduces COCO error by 97.76%, ending only 0.53% above the original warmup pair. However, its PushT image error becomes 8.72 times the task-adapted reference. Preserve the original task decoder and use the recovered decoder separately for generic visualization. Encoder and pose-head parameters are exactly unchanged; this experiment does not improve pose estimation, memory, prediction or control. The decoder-refit checkpoint is not a silently interchangeable controller dependency.

The matched fresh decoder has 26.33% higher COCO error on the adapted encoder than on the warmup encoder. That residual gap can reflect representation changes and/or optimization difficulty within the fixed budget; it is not proof of irretrievable information loss. More training, added capacity, skips and registers are not established requirements by these results.

## Protocol and uncertainty

The [prospective protocol](decoder-recovery-plan-2026-09-08.md) used three arms, each 2,000 AdamW updates, batch 128, learning rate 0.0003, weight decay 0.0001 and clipping 1.0. Seed 5107, identical sampled COCO training streams and identical initial fresh decoders provide matched controls. Each arm presented 256,000 images; training plus validation took approximately 88–93 seconds per arm on the RTX 3050. Selection used fixed 2,048-image COCO validation samples every 100 updates. All selected checkpoints were the final step 2,000. No PushT images entered refitting.

Evaluation used the same 4,146 COCO and 2,506 PushT test frames for every checkpoint, including both original references and selected/final refits. These previously inspected internal test populations and one training seed make this exploratory evidence, not independent confirmation. Train-only mean-image baselines are 0.069516896 for COCO and 0.003226755 for PushT. The fresh decoder on the adapted encoder is worse than the PushT mean baseline despite recovering COCO.

Paired bootstrap intervals use 2,000 whole-source-group resamples with frame-weighted means: 4,132 COCO groups and 22 PushT groups. The COCO refit-minus-original-warmup difference is 0.000032336, with 95% interval [0.000016937, 0.000047727]. The fresh-decoder adapted-minus-warmup difference is 0.001331612 [0.001310376, 0.001353353]. PushT refit-minus-task-reference is 0.001447505 [0.001375932, 0.001531628]. These intervals describe sampled groups conditional on these trained models, not variability across training seeds.

An initial GPU evaluation used default cuDNN TF32 settings. Its complete outputs were archived under `.runtime/decoder_recovery_default_gpu_math`. The official evaluation was rerun with CUDA matmul TF32 and cuDNN TF32 explicitly disabled, without retraining; reference numbers now agree with the earlier CPU results within rounding. The runtime settings, exact checkpoint hashes, group draws and per-frame errors are preserved in the raw ledger.

## Figures and evidence

[Verified interactive dashboard](../runs/experiment_dashboard.html) · [Raw metrics and intervals](../runs/decoder_recovery_2026-09-08/evaluation/metrics.json) · [Matching audit](../runs/decoder_recovery_2026-09-08/evaluation/matching_audit.json) · [Prior curriculum dashboard](../runs/decoder_recovery_2026-09-08/prior_reports/experiment_dashboard.html)

![COCO recovery on six fixed examples](../runs/decoder_recovery_2026-09-08/evaluation/coco_recovery.png)

![PushT retention on six fixed examples](../runs/decoder_recovery_2026-09-08/evaluation/pusht_recovery.png)

![Matched validation learning curves](../runs/decoder_recovery_2026-09-08/evaluation/learning_curves.png)

All 359 software tests pass, including the three installed-browser checks. The dashboard passes package, source-interaction and desktop/mobile browser verification. Exact frozen E/H state, matched sample streams, equal fresh-decoder initialization and immutable parent hashes pass. The warmup reference had no trained pose head; its refit arm's unused head is a placeholder, not a pretrained readout.

## Broader world-model implications

Claude completed the requested public literature review; the [reviewed findings and primary sources](world-model-literature-2026-09-08.md) retain the earlier domain-switch, geometry/readout, generic-retention, memory and causal-prediction points. Existing methods justify testing decoder/readout refitting, head-first adaptation, replay and separation of representation from dynamics. They do not establish a ready-made fix for this model.

For geometry, the proposed next comparison is a spatial pose head with location heatmaps or orientation keypoints, consistent transforms and unseen-configuration evaluation. For memory, first compare richer frozen-state readouts before changing recurrence; position and velocity supervision already exist. For prediction, test multi-step physical objectives through verified readouts, since the current predictor trains on latent error while physical readout error is diagnostic. These remain proposals with new bounded protocols required before execution.

Task/domain conditioning and mixed reconstruction training could address the observed decoder retention tradeoff. Reconstruction-versus-prediction conditioning is distinct from domain conditioning: a state-to-image decoder can serve either operation if the predictor supplies the future state. Skips must remain causal and should be checked for copying static appearance while missing motion. ViT register tokens are per-image scratch space; persistent temporal memory already exists here. Neither extra encoder registers nor decoder memory was tested or adopted. All authorized recovery training and evaluation are complete; no additional training is queued.
