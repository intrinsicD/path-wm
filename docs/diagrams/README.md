# Architecture drawings

[Open the architecture atlas](../architecture-atlas.html) for thirteen expandable
views from the agent loop to individual attention blocks. The same drawings have
[editable Mermaid source and notes](../architecture-atlas.md) and full-size SVGs
in `atlas/`.

`architecture-atlas.json` is the manually audited source: nodes, labelled edges,
component roles, implementation references and scope notes. It describes the
categorical agent, the Gaussian photo path and the controlled entity/planning
experiments separately. Dashed proposed graph/DAG structures are not claimed as
implemented or trained. Feedback-edge annotations only influence layout.

The overview's colors now track **discussion coverage**: red = to discuss, blue =
discussed. This overrides role colors for diagram1 only; diagrams2–13 retain their
existing role legend. `graphs[0].discussion.nodes` records status and conversation
evidence for every overview node. Update those records after substantive discussion
and regenerate; [the generated checklist](../architecture-discussion.md) explains
the initial assessment and remaining questions. Coverage is not readiness or quality.

Regenerate offline with Node.js and `@viz-js/viz` (used version 3.25.0):

```bash
node docs/diagrams/render_architecture_atlas.mjs
# Or supply an already installed package's dist/viz.js as the sole argument.
```

The renderer checks graph endpoints, source files and named source symbols. It
generates SVGs, Markdown and a self-contained local HTML document. It does not
import the model, train, run evaluations or contact an external service. Source
references describe the recorded code snapshot; revise them when architecture
changes. The older `architecture.*`, `data_flow.*` and modality diagrams remain
the separately executed Gaussian reference exports, preserved unchanged.

Validation on 14 September 2026: 13 graphs, 156 nodes and 199 edges render without
Graphviz warnings; all local links/anchors resolve; 384 HTML IDs are unique;
SVG node/edge counts match the source, and the page script passes syntax checking.
The drawings were rasterized for visual inspection. Browser URL policy blocked
automated access to the local page, so browser interactions, responsive layout and
printing are not claimed verified. Native expandable sections and the SVG/Markdown
exports provide access independently of the page's optional convenience controls.
