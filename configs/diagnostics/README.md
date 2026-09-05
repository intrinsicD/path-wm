# Authorized short checks

User approved this sequence on 2026-09-05: replay alignment; ten identical goals
through upstream and local evaluators with released weights; a small fixed-data
batch-128 learning check capped at 15 minutes; a second-dataset check.
Architecture and objective stay fixed. No long training or research extensions.

At the first check, official archives were downloading. Complete early episodes extracted
from downloaded source byte ranges may support interim checks. Their source
receipts must identify the subset and incomplete full-archive verification.
Subset-fitted action statistics and early correlated episodes cannot establish
paper benchmark performance. Compare evaluators on identical cases, statistics,
CEM samples/seeds, source goal images, and reset semantics.

Pass/fail interpretation: recorded transition alignment must be explained;
upstream/local disagreement indicates integration. Agreement and positive
control success permit a capped training diagnosis. Training-set loss alone is
insufficient: report persistence/shuffled-action controls, representation spread,
open-loop error, and train/eval BatchNorm behavior. Report all failed checks.

Follow-up authorization: user accepted the next bounded sequence after the first
report. `pusht_cached_learning.yaml` repeats the same random initialization,
split, batch, architecture, loss and 400-update schedule with cached data and
zero workers, retaining the 840-second limit. Evaluate the saved checkpoint
and diagnostic clones separately, plus a few held-out control starts. Complete
source verification and released-checkpoint reference evaluation as downloads
finish. This does not authorize long training or component research.


Recovery follow-up complete: both full source archives pass their pinned hashes
and are extracted. The released checkpoint reaches 45/50 full-source goals;
the untouched cached 400-update checkpoint reaches 1/5 related held-out prefix
goals. See [the current report](../../docs/reference-validation.md). No long
training or component research has been launched.
