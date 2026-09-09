# Critical collaboration with Claude

Alex adopted this workflow after the earlier encoder design collaboration. Keep
its useful discipline; no task-specific coordinator is required.

1. **Bound the review.** State the decision, constraints, alternatives, uncertainty
   and the specific criticism needed. The current export boundary is public-only:
   do not send private code, datasets, measurements or unrelated history externally.
2. **Use actual Claude.** Prefer a reachable Claude interface or the installed CLI
   with explicitly scoped tools. Do not substitute another model under its name.
   Work independently while the review runs. Never bypass a denied action.
3. **Verify.** Check claims against local code/evidence or public primary sources.
   Distinguish findings, hypotheses and preferences. Agreement is not validation.
4. **Reconcile.** Return concrete corrections; obtain acknowledgment of withdrawn
   claims and name remaining differences. More rounds need a decision to resolve.
5. **Keep a compact record.** Save exact briefs, responses and execution receipts
   under an ignored `runs/reviews/<name>/` directory. Put material corrections and
   adopted decisions in the current plan; avoid another permanent protocol stack.
   CLI cost fields are API-equivalent estimates, not subscription bills.
6. **Test locally.** Essential implementation and experimental checks remain the
   implementing agent's responsibility. Claude cannot authorize scope or budgets.

Use for consequential scientific choices and substantive review, not routine
formatting or every small edit. The prior workflows and examples are recoverable
from `archive/pre-modular-2026-09-09`.
