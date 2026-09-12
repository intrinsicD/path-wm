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
