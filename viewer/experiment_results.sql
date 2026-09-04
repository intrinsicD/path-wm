-- Reproducible DuckDB source view for the experiment dashboard.
-- viewer/dashboard.py performs the same file discovery with Python's standard JSON/YAML readers so
-- experiment execution does not acquire a database dependency. These queries provide a compact audit
-- path over the immutable run ledger and are the canonical source reference embedded in the dashboard.

CREATE OR REPLACE VIEW path_wm_evaluation_records AS
SELECT *
FROM read_json_auto(
    'runs/**/metrics.json',
    filename = true,
    union_by_name = true
);

CREATE OR REPLACE VIEW path_wm_run_summaries AS
SELECT *
FROM read_json_auto(
    'runs/**/run_summary.json',
    filename = true,
    union_by_name = true
);

CREATE OR REPLACE VIEW path_wm_training_log AS
SELECT *
FROM read_json_auto(
    'runs/**/training.jsonl',
    filename = true,
    format = 'newline_delimited',
    union_by_name = true
);
