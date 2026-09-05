-- Executed with SQLite JSON1 over :reconciled_runs (JSON RunResult records).
-- viewer.ledger validates raw JSON/JSONL first; viewer.dashboard binds the records.
-- Chart input trajectories are deterministically sampled to at most 50 rows;
-- exact validation rows and raw source paths remain in the bound records.
-- dataset: inventory
SELECT json_extract(value, '$.label') AS run,
       json_extract(value, '$.kind') AS kind,
       json_extract(value, '$.status') AS status,
       COALESCE(CAST(json_extract(value, '$.step') AS TEXT), 'Not recorded') AS step,
       COALESCE(json_extract(value, '$.context.baseline_gate'), 'Not recorded') AS gate,
       (SELECT group_concat(value, ', ') FROM json_each(json_extract(r.value, '$.source_paths'))) AS sources
FROM json_each(:reconciled_runs) AS r;
-- dataset: training_runs
SELECT json_extract(value, '$.label') AS run FROM json_each(:reconciled_runs)
WHERE json_extract(value, '$.kind') = 'training';
-- dataset: training_loss
SELECT json_extract(r.value, '$.label') AS run, json_extract(t.value, '$.step') AS step,
       m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r,
     json_each(json_extract(r.value, '$.sampled_training')) AS t, json_each(t.value) AS m
WHERE m.key IN ('loss', 'pred_loss', 'sigreg_loss');
-- dataset: validation
SELECT json_extract(r.value, '$.label') AS run, json_extract(t.value, '$.step') AS step,
       m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r,
     json_each(json_extract(r.value, '$.sampled_validation')) AS t, json_each(t.value) AS m
WHERE m.key IN ('pred_mse', 'identity_mse', 'shuffled_action_mse', 'zero_action_mse');
-- dataset: embedding
SELECT json_extract(r.value, '$.label') AS run, json_extract(t.value, '$.step') AS step,
       json_extract(t.value, '$.embedding_std') AS value
FROM json_each(:reconciled_runs) AS r,
     json_each(json_extract(r.value, '$.sampled_validation')) AS t
WHERE json_type(t.value, '$.embedding_std') IS NOT NULL;
-- dataset: prediction_runs
SELECT json_extract(value, '$.label') AS run FROM json_each(:reconciled_runs)
WHERE json_extract(value, '$.kind') = 'prediction';
-- dataset: prediction
SELECT json_extract(r.value, '$.label') AS run, m.key AS metric, m.value AS value,
       json_extract(r.value, '$.step') AS step, json_extract(r.value, '$.metrics.examples') AS examples
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.metrics')) AS m
WHERE json_extract(r.value, '$.kind') = 'prediction'
  AND m.key IN ('pred_mse', 'identity_mse', 'shuffled_action_mse', 'zero_action_mse');
-- dataset: control
SELECT json_extract(value, '$.label') AS run,
       replace(CASE WHEN json_extract(value, '$.label') LIKE 'diagnostics/%'
                    THEN substr(json_extract(value, '$.label'), 13)
                    ELSE json_extract(value, '$.label') END, '_', ' ') AS label,
       json_extract(value, '$.context.case_set') AS case_set,
       json_extract(value, '$.metrics.successes') AS successes,
       json_extract(value, '$.metrics.cases') AS cases,
       json_extract(value, '$.metrics.success_rate') AS success_rate,
       COALESCE(json_extract(value, '$.context.protocol'), 'Not recorded; consult sources') AS protocol,
       json_extract(value, '$.status') AS status
FROM json_each(:reconciled_runs) WHERE json_extract(value, '$.kind') = 'control';
-- dataset: coverage
SELECT count(CASE WHEN json_extract(value, '$.kind') = 'training' THEN 1 END) AS training_runs,
       count(CASE WHEN json_extract(value, '$.kind') = 'training'
                   AND json_extract(value, '$.status') = 'complete' THEN 1 END) AS completed_training,
       count(CASE WHEN json_extract(value, '$.kind') = 'control' THEN 1 END) AS control_evaluations,
       count(CASE WHEN json_extract(value, '$.kind') = 'prediction' THEN 1 END) AS prediction_evaluations
FROM json_each(:reconciled_runs);
-- dataset: context
SELECT json_extract(r.value, '$.label') AS run, m.key AS field, m.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.context')) AS m;
-- dataset: metrics
SELECT json_extract(r.value, '$.label') AS run, 'result' AS section, m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.metrics')) AS m
UNION ALL
SELECT json_extract(r.value, '$.label') AS run, 'last logged training' AS section, m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.training[#-1]')) AS m
UNION ALL
SELECT json_extract(r.value, '$.label') AS run,
       'validation at step ' || json_extract(t.value, '$.step') AS section, m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.validation')) AS t,
     json_each(t.value) AS m;
