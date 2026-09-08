# Critical collaboration with Claude

The user explicitly adopted this reusable workflow on 8 September 2026 after
the [encoder design discussion](encoder-depth-scale-cowork-2026-09-08.md).
Use it for consequential architecture choices, experiment design, uncertain
scientific interpretations and substantive implementation review. Routine edits
do not need a second model. This standing preference authorizes appropriately
scoped Claude consultation again; do not ask the user to repeat it.

1. **Prepare an evidence-first brief.** State the decision, exact relevant code
   behavior, measured results and their limits, constraints, candidate protocol
   and rejected alternatives. Ask for independent criticism, counterexamples,
   confounds and an actionable recommendation. Share only the relevant brief or
   code excerpts, without credentials, private datasets or unrelated files.
2. **Work independently.** While Claude reviews, inspect code, primary literature
   or raw results and implement already-authorized independent work. Prefer a
   reachable Claude agent; if none is available, use the installed Claude CLI
   with explicit tool access. Do not substitute a different model while calling
   it Claude. No tool path may bypass a denied action.
3. **Verify before adopting.** Check claims against current implementation,
   primary sources and actual experiment records. Separate observed facts,
   literature results, hypotheses and preferences. Agreement between models is
   not empirical validation, and Claude cannot authorize scope changes.
4. **Reconcile concrete disagreements.** Send corrections and a revised proposal
   back. Require explicit acknowledgment of withdrawn claims and identify any
   remaining uncertainty. Use further rounds only when they resolve a decision.
5. **Preserve the record.** Save the exact brief, replies and execution receipts
   under the experiment's ignored `runs/` directory. Put the adopted decision,
   attribution, important corrections, remaining differences and artifact paths
   in the tracked protocol/report. Record CLI cost fields as API-equivalent
   estimates, never as subscription bills.
6. **Test the decision.** Follow the ordinary plan/test/development/comparison
   workflow. Have Claude inspect scientific invariants and interpretations where
   useful, while the implementing agent remains responsible for verification.

The initial successful example used three exchanges: independent review,
code-grounded challenge, and final reconciliation. It corrected claims about
already-present frozen targets and multi-step training, unsupported quantization
limits, and the interpretation of a failed pretrained comparator. Preserve this
critical method rather than seeking consensus for its own sake.

The reference CLI runner is
`runs/architecture_cowork_2026-09-08/run_claude.py`. Its safe-mode session used only
WebSearch/WebFetch, explicit empty MCP configuration and a supplied prompt,
preserving replies and receipts. Adapt tool access to the actual bounded task;
do not grant broad repository execution for a literature-only review.
