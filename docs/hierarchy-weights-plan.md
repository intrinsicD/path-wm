# Directly constructed hierarchy weights — 12 September 2026

The user asks for numerical weight construction and another test of the residual
hierarchy. The preceding hierarchy comparison used ordinary initialization and
optimizer training. Preserve those files. This experiment writes all parameters
of the same deep_fusion model (width32, three scales, two blocks per scale and two
final fusion blocks, unchanged RGB/mask DenseHeads), with zero optimizer updates.

## Frozen candidates and population

Evaluate ordinary untrained seeds7401/7402, one deterministic handwritten candidate,
and one copy with only the final RGB/mask convolutions fitted on training data.
Handwritten weights overwrite every random parameter, so duplicate seeds are not
independent constructed candidates. No selection among candidates; report all four.
Use the same512 train/128 validation/128 test COCO64 rows and masks as the prior
screen. These test images and previous scores are already known: this is a reused
internal comparison, not an independent generalization estimate. No test-guided
weight or threshold adjustment. Software fixtures use unrelated synthetic pixels.

Hand assignment: zero every parameter then set normalization gains; patch channels
0..5 encode paired signed RGB means with gain256 and center0.5; channels6/7 are
constant +4096/-4096 carriers. Keep positional computations in the original forward.
Each transformer attention uses identity Q/K/V and output gain8; paired GELU MLP
weights implement0.01 times the normalized input. This retains nonzero processing
and fusion while making bounded corrections to the large carrier/color features.
Scale and modality learned embeddings and condition modulation are zero.

Decoder projections read paired normalized RGB channels with coefficient2; combine
scales with weights0.80/0.15/0.05. Each convolution transports signed color pairs;
constant +/-64 carriers per normalization group stabilize scale. Normalization gains
match the carrier-only standard deviation. RGB branch adds4 before each GELU to
approximate linear transport, then predicts sigmoid(4*(color-0.5)). The mask branch's
last GELU uses zero offset; its final readout is4 times the summed signed-pair GELU
responses minus0.5. This is a color/intensity-distance heuristic, not object semantics.
Constants are analytical guesses, never called pretrained or data-free learning.

The fitted candidate changes only the two final1x1 convolutions:132 coefficients
including biases. Streaming float64 ridge regression uses all valid training mask
pixels and all training RGB pixels, target logit(clamp(RGB,0.01,0.99)) and signed
mask targets +/-2. Regularizer0.001 on averaged feature Gram, bias unpenalized.
This is supervised, closed-form fitting despite zero backward/optimizer calls.

## Evidence and criteria

Each candidate gets standard Run metadata/checkpoint/report plus strict-loadable
weights.pt, numerical construction/fit receipt, raw test_outputs.npz and metrics.
Reuse RGB MSE, valid-frame foreground IoU/BCE, observed/shuffled/zero-feature controls
and gray/full/empty references. Compare with the saved trained deep_fusion runs;
different training exposure is explicit. Main useful-perception gate: IoU exceeds
the full-foreground reference by0.02, RGB MSE below gray, and observed features beat
both shuffled and zero on RGB MSE and mask IoU. Separately report whether a candidate
beats each trained reference by IoU0.02 without RGB MSE worsening over5%.
No criterion is evidence of persistent identity, speech, memory or video capability.

Essential RED tests: complete deterministic assignment and binary reload; nonzero
feature processing/input-sensitive RGB through the unchanged architecture; fitting
can change only final readouts and cannot call backward/optimizer; synthetic run
and cached resume preserve binary/raw-output hashes. Commit tests/plan, implement,
run appropriate/full CPU checks, then commit before real evaluation. Inspect the
synthetic output and real report examples. GPU cap4GiB, leave1GiB free, at most300s
per candidate and20min total. No downloads, webcam, shared-source edits during runs,
or replacement of existing checkpoints. Reports use the existing renderer and carry
structural-only status while browser access remains blocked.

Claude review was attempted with a bounded conceptual brief. Automatic approval
review rejected the external send as containing nonpublic architecture/experiment
details; no response was received and no alternate route is used. Local review and
tests proceed. Brief: runs/reviews/continuation_2026-09-11/constructed-hierarchy-brief.txt.

## Implementation checks before real evaluation

Two initial checks failed on the missing constructor/fitter, then passed. Added a
synthetic end-to-end run, strict reload, cached resume and corruption rejection with
backward/optimizer calls forbidden. Its initial fixture passed full-population labels
to a subset API; fixed the fixture alignment without changing the data implementation.
All197 CPU tests pass in173.09s. The unchanged hierarchy remains active (fusion
ablation changes features); primary-color inputs yield the correct dominant output
channel with over0.5 separation. Same hand assignment has the same state hash from
different random seeds. Ridge fitting changes exactly the four declared final-layer
tensors. GPU forward preflight reserves112MiB, with no backward call.

Implementation is in the existing `experiments/hierarchy_fusion.py` recipe through
`--weight-method untrained|constructed|ridge`; default `trained` preserves the ordinary
experiment path. Formal real-image evaluation starts only after this source is committed.

## Completed comparison

Source3d19e17. Four sequential GPU evaluations finish in25.91s total, including
process setup. All retain the same1,797,028-parameter architecture and use zero
optimizer updates/backward calls. Pure handwritten weights contain4,843 nonzero
parameters; the fitted variant contains4,930, with132 final-readout coefficients
fitted from512 training images. Each binary is about7.3MB in ordinary PyTorch format.

| Weights | RGB MSE | Foreground IoU | Mask BCE |
|---|---:|---:|---:|
| Untrained7401 | 0.079823 | 0.257903 | 0.697517 |
| Untrained7402 | 0.078056 | 0.093759 | 0.716319 |
| Fully handwritten | 0.013853 | 0.204784 | 0.714076 |
| Handwritten + fitted readout | 0.013693 | 0.072054 | 0.587418 |
| Previous trained7401,384 updates | 0.010849 | 0.257652 | 0.560581 |
| Previous trained7402,384 updates | 0.011635 | 0.238005 | 0.548026 |

Both constructed candidates reconstruct input-dependent patch colors, but neither
beats either trained reference on the paired RGB/IoU criterion. Both fail the
useful-perception gate: full foreground reaches IoU0.321622. The fitted readout
improves BCE while reducing IoU at the frozen threshold; it marks only4.32% of valid
test pixels foreground, versus38.13% for the handwritten heuristic. No threshold or
weight adjustment was made after test access. Neither result supports object identity,
semantic understanding, useful segmentation or general pretrained-weight generation.

Inference/fitting GPU peaks120–136MiB reserved; at least4.60GiB device memory remains
free. The earlier112MiB value is the separate constructed forward preflight. All197
CPU tests pass; independent Torch reconstruction of36 metrics agrees within2.3e-16.
Original eight trained checkpoint/output hashes remain unchanged. Training row hashes,
strict binary/checkpoint equality and completed GPU cached resume pass. Resume leaves
weights, checkpoint, raw outputs, metrics and report hashes unchanged and does no fit
or score. Six preselected validation comparisons were visually inspected: color/block
transport is visible, while mask probabilities do not delineate objects adequately.
HTML verification is structural-only; the report was queued in the Codex file panel.

[Report](../runs/hierarchy_weights_v1/report.html),
[comparison and verification](../runs/hierarchy_weights_v1/comparison.json),
[resume checks](../runs/hierarchy_weights_v1/resume_check.json).
Claude review remains blocked by the automatic approval rejection recorded above.

Generate an independent copy or load the completed files:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.hierarchy_fusion \
  --arm deep_fusion --weight-method constructed --device cuda \
  --output runs/my_constructed_hierarchy
# --weight-method ridge additionally fits the final132 coefficients on training data.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.hierarchy_fusion \
  --resume runs/hierarchy_weights_v1/ridge_7401
```

```python
import torch
from experiments.hierarchy_fusion import build_model

model, _ = build_model(7401, "deep_fusion")
payload = torch.load(
    "runs/hierarchy_weights_v1/constructed_7401/weights.pt", weights_only=True
)
model.load_state_dict(payload["model"], strict=True)
model.eval()
# model(rgb): RGB floats[B,3,64,64] in[0,1], returns RGB and foreground-mask logits.
# ridge_7401/weights.pt loads identically; its final readout is supervised/data-fitted.
```
