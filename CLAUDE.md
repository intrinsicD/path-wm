# CLAUDE.md — how we work on PATH-WM

Read in this order at the start of a session:

1. [Experiment and development workflow](docs/experiment-workflow.md): standing
   process, essential tests, commits, scientific integrity and mandatory HTML.
2. [Current project state](docs/project-state.md): implementation objective,
   active experiment, evidence, constraints and already-authorized next steps.
3. The active experiment's configuration, protocol and result report linked there.

Keep this entry point and the workflow independent of any particular model,
dataset, architecture or experiment target. Update project state when the goal
changes; do not replace the harness. Explicit user instructions take precedence.

Every completed experiment run must refresh and verify
[`runs/experiment_dashboard.html`](runs/experiment_dashboard.html) from its raw
ledger, with charts and exact values available for visual inspection. A raw run
can succeed while reporting fails; report that failure visibly and repair the
HTML before calling the experiment workflow complete.

The standing harness was recovered from Git commit `eca742a` and adapted to the
current ledger format.
