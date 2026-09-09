# Current work

**Active:** user-authorized modular restart, 9 September 2026.
[Migration plan](migration.md) defines keep/remove choices and acceptance checks.

Build a small `pathwm` library and editable perception recipe first. Verify against
the preserved source, then migrate useful memory/prediction pieces, simplify docs
and packaging, and remove retired active machinery. A runnable short sequence
recipe follows the verified perception path. Development runs only; no long model
training or new model-quality claim is authorized by this refactor.

Reference source: `archive/pre-modular-2026-09-09` at
`e95b6a6252cae72402de0dd93f419e8e93d25d12`. Data and completed runs remain on disk.
The old dashboard is a historical artifact. New reports belong to individual runs.

Status: plan written; essential contract tests fail as expected because the new
implementation does not yet exist. Workflow adapted to prevent renewed sprawl.
