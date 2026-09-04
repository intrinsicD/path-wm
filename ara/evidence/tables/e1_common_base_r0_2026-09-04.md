# E1 common-base real-A/V R0 development evidence — 2026-09-04

Status: development plumbing only. These runs use the official 20-clip TAU example bundle and 20 optimizer
steps; they are not promotion evidence for the representation architecture.

## Fixed setup

| Field | Value |
|---|---|
| Corpus | TAU Urban Audio-Visual Scenes 2021 development examples, DOI 10.5281/zenodo.4477542 |
| Source integrity | official examples/meta MD5 verified; every source and normalized shard SHA-256 recorded |
| Ingested cohort | 20 synchronized 10-second clips; 9 official train, 11 official eval |
| Normalization | video 8 fps, RGB uint8 64×64; audio mono float32 16 kHz |
| Window | 0.5 seconds current; +0.5 seconds future; +2.0 seconds R1 shifted control |
| R0 sampling | video/audio independently sampled; labels absent from `RepresentationBatch` |
| Training | seed 0, CUDA, AdamW, batch 2, 20 steps, bf16 activations |
| Panel | 2 fixed held-out batches / 4 examples, independent evaluation RNG |
| Gate | rank fraction ≥0.25; feature std ≥0.1; masked/future advantage >0; temporal margin >0 per modality |

The initial held-out panel was bit-for-bit identical for all three specs: video/audio rank fractions
0.126192/0.053324, feature standard deviations 0.717280/0.723665, and negative masked/future prediction
advantages in both modalities.

## Matched final panels

| Spec | Covariance weight | Video rank | Audio rank | Video std | Audio std | Video masked/future advantage | Audio masked/future advantage | Video/audio temporal margin | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `common_base.yaml` | off | 0.107513 | 0.043458 | 0.755219 | 0.726758 | +0.160144 / +0.216672 | -0.004970 / -0.014114 | +0.042317 / +0.002875 | fail (4) |
| `common_base_rank.yaml` | 0.01 | 0.114587 | 0.046985 | 0.642119 | 0.618713 | +0.135694 / +0.158080 | -0.011079 / -0.026483 | +0.030274 / +0.002243 | fail (4) |
| `common_base_rank_balanced.yaml` | 0.002 | 0.109685 | 0.044552 | 0.731594 | 0.702014 | +0.155324 / +0.204035 | -0.006122 / -0.017100 | +0.037914 / +0.002721 | fail (4) |

All runs failed the same four frozen conditions: video and audio effective rank below 0.25, plus audio
masked and future prediction advantage below zero. Video prediction and both temporal-retrieval margins
remained positive.

## Objective-scale audit

On the exact seed-0 initial training batch, unweighted encoder-plus-adapter gradient norms were:

| Component | Raw loss | Gradient norm | Configured weighted norm in first covariance trial |
|---|---:|---:|---:|
| masked latent | 2.017569 | 1.007064 | 1.007064 |
| future latent | 2.021843 | 1.349980 | 1.349980 |
| variance | 0.299651 | 0.578425 | 0.057842 (`weight=0.1`) |
| covariance | 8.060951 | 27.903984 | 0.279040 (`weight=0.01`) |

Thus weight 0.01 made covariance's weighted gradient 4.8 times the variance-floor gradient and exposed the
shrinkage route. Weight 0.002 was selected from this initialization audit, before its held-out result,
giving a weighted covariance norm of approximately 0.055808.

## Provenance and limits

- Data boundary and red tests: commits `0486841`, `93eefaa`.
- Real ingestion/trainer/panel: commit `def60b2`; local ledger `runs/dev/common_base/0/`.
- Covariance guardrail and controlled specs: commit `e5d7018`; local ledgers
  `runs/dev/common_base_rank/0/` and `runs/dev/common_base_rank_balanced/0/`.
- Fast suite at checkpoint: 154 passed, 2 opt-in tests deselected, one pre-existing tensor-scalar warning.
- Dashboard packaging completed with `structural_only` verification; raw JSON/YAML/JSONL ledgers remain
  authoritative.
- Next decision requires the complete 34-hour corpus, matched seeds/budget, and the unchanged held-out gate.
