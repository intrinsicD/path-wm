# P1-T: crop-aware category accessibility

Authorized conditional continuation of the overnight encoder/decoder investigation.
No albedo, new downloads, software-use labels or controller claim.

Question: do these fixed CNN/ViT packages make generic category presence accessible
to a separately trained pooled readout? This complements spatial RGB, foreground
and PushT geometry; it is not a localization or open-vocabulary test.

Use exactly the existing4,096/512/512 COCO mask-subset identities and the canonical
RGB64 crop. A category is positive if at least one pixel of its decoded non-crowd
mask survives nearest-neighbor label resizing and the actual crop. A visible crowd
of that category makes an otherwise negative label unknown. A visible non-crowd
positive overrides unknown. Unknowns are excluded from both loss and AP. Preserve
full80-category labels; train/report classes with at least20 positive and20 known
negative training images. No validation/test support controls class inclusion.
Report absent-positive/absent-negative held-out class coverage separately.

The head independently LayerNorms each native source token, mean-pools each level,
concatenates the two vectors, projects to128, applies GELU and outputs category
logits. Native ViT's derived coarse grid does not add independent evidence. No
spatial or RGB head is updated. Shape-compatible common initialization is paired;
width-dependent input projection uses a separate RNG stream.

Seeds9107/9108/9109;2,000updates, batch64; AdamWlr3e-4, wd1e-4, clip1; FP32/TF32off.
Unweighted BCE averages known class-image entries. Validate every100updates; select
maximum macro average precision, earliest exact tie. AP is the stepwise
precision-recall integral, grouping identical scores at a shared threshold;
unknown rows and classes with no positives are excluded. Report comparable class
coverage, per-class AP/support and the training-prevalence constant-score baseline.
The test split does not select checkpoints or class support. Preserve endpoint
as well as selected results, raw scores/labels/known masks and paired sampler hashes.

No numerical semantic pass threshold is invented. Report paired macro-AP deltas
and per-class consistency; the question is relative accessibility under this head
and data budget. It does not establish general representation adequacy.

Budgets: preparation at most30minutes, development at most15minutes, each formal
fit at most10minutes (six fits≤60fitting minutes), evaluation/reporting≤30minutes.
The global06:00Berlin training cutoff and07:00 report deadline still apply. Label
preparation and CPU checks may run while P1 trains; GPU training waits until the
P1 coordinator releases the GPU. Freeze source/data/seed identities before formal
P1-T training. Essential tests: crop exclusion, crowd unknown handling and AP tie
invariance. First run an explicit small development configuration and verified HTML.
