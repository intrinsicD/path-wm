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
WHERE m.key IN ('pred_mse', 'identity_mse', 'shuffled_action_mse', 'zero_action_mse', 'rollout_mse');
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
       json_extract(value, '$.metrics.initial_successes') AS initial_successes,
       json_extract(value, '$.metrics.noninitial_cases') AS noninitial_cases,
       json_extract(value, '$.metrics.noninitial_successes') AS noninitial_successes,
       json_extract(value, '$.metrics.noninitial_success_rate') AS noninitial_success_rate,
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
-- dataset: ranking_runs
SELECT json_extract(value, '$.label') AS run FROM json_each(:reconciled_runs)
WHERE json_extract(value, '$.kind') = 'ranking';
-- dataset: ranking
SELECT json_extract(r.value, '$.label') AS run,
       json_extract(c.value, '$.candidate') AS candidate,
       CASE WHEN json_extract(c.value, '$.candidate') LIKE 'random_%' THEN 'random'
            ELSE json_extract(c.value, '$.candidate') END AS family,
       json_extract(c.value, '$.predicted_cost') AS predicted_cost,
       json_extract(c.value, '$.position_error') AS position_error,
       json_extract(c.value, '$.angle_error') AS angle_error,
       json_extract(c.value, '$.success_terminal') AS success_terminal,
       json_extract(c.value, '$.actual_latent_cost') AS actual_latent_cost,
       json_extract(c.value, '$.initial_source_sim_mse') AS initial_source_sim_mse
FROM json_each(:reconciled_runs) r, json_each(json_extract(r.value, '$.ranking')) c;
-- dataset: rollout_error
SELECT json_extract(r.value, '$.label') AS run, json_extract(c.value, '$.candidate') AS candidate,
       (CAST(t.key AS INTEGER)+1)*5 AS environment_step,
       replace(m.key, '_by_step', '') AS metric, t.value AS value
FROM json_each(:reconciled_runs) r, json_each(json_extract(r.value, '$.ranking')) c,
     json_each(c.value) m, json_each(m.value) t
WHERE m.key IN ('rollout_mse_by_step', 'copy_mse_by_step')
  AND json_extract(c.value, '$.candidate') NOT LIKE 'random_%';
-- dataset: rollout_goal
SELECT json_extract(r.value, '$.label') AS run, json_extract(c.value, '$.candidate') AS candidate,
       (CAST(t.key AS INTEGER)+1)*5 AS environment_step,
       replace(m.key, '_by_step', '') AS metric, t.value AS value
FROM json_each(:reconciled_runs) r, json_each(json_extract(r.value, '$.ranking')) c,
     json_each(c.value) m, json_each(m.value) t
WHERE m.key IN ('predicted_cost_by_step', 'actual_cost_by_step')
  AND json_extract(c.value, '$.candidate') NOT LIKE 'random_%';
-- dataset: training_scalar
-- viewer.dashboard splits these rows into one train_<metric> dataset per panel.
SELECT json_extract(r.value, '$.label') AS run, json_extract(t.value, '$.step') AS step,
       m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r,
     json_each(json_extract(r.value, '$.sampled_training')) AS t, json_each(t.value) AS m
WHERE m.key IN ('loss', 'pred_loss', 'sigreg_loss', 'grad_norm', 'lr');
-- dataset: validation_ratio
-- Scale-free ratios within one checkpoint: 1.0 means no better than the control.
SELECT run, step, metric, value FROM (
  SELECT json_extract(r.value, '$.label') AS run, json_extract(t.value, '$.step') AS step,
         'prediction / copy' AS metric,
         1.0 * json_extract(t.value, '$.pred_mse') / json_extract(t.value, '$.identity_mse') AS value
  FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.sampled_validation')) AS t
  WHERE json_extract(t.value, '$.identity_mse') > 0
  UNION ALL
  SELECT json_extract(r.value, '$.label'), json_extract(t.value, '$.step'), 'prediction / shuffled actions',
         1.0 * json_extract(t.value, '$.pred_mse') / json_extract(t.value, '$.shuffled_action_mse')
  FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.sampled_validation')) AS t
  WHERE json_extract(t.value, '$.shuffled_action_mse') > 0
  UNION ALL
  SELECT json_extract(r.value, '$.label'), json_extract(t.value, '$.step'), 'prediction / zero actions',
         1.0 * json_extract(t.value, '$.pred_mse') / json_extract(t.value, '$.zero_action_mse')
  FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.sampled_validation')) AS t
  WHERE json_extract(t.value, '$.zero_action_mse') > 0
  UNION ALL
  SELECT json_extract(r.value, '$.label'), json_extract(t.value, '$.step'), 'rollout / one-step copy',
         1.0 * json_extract(t.value, '$.rollout_mse') / json_extract(t.value, '$.identity_mse')
  FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.sampled_validation')) AS t
  WHERE json_extract(t.value, '$.identity_mse') > 0 AND json_type(t.value, '$.rollout_mse') IS NOT NULL
  UNION ALL
  SELECT json_extract(r.value, '$.label'), json_extract(t.value, '$.step'), 'action effect / prediction',
         1.0 * json_extract(t.value, '$.action_effect') / json_extract(t.value, '$.pred_mse')
  FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.sampled_validation')) AS t
  WHERE json_extract(t.value, '$.pred_mse') > 0 AND json_type(t.value, '$.action_effect') IS NOT NULL
) ORDER BY run, metric, step;
-- dataset: internals_scalar
-- Checkpoint-inspection scalars for records with a training step; viewer.dashboard groups them into panels.
SELECT json_extract(r.value, '$.label') AS run, json_extract(r.value, '$.internals.family') AS family,
       json_extract(r.value, '$.internals.training_run') AS training_run,
       json_extract(r.value, '$.step') AS step, m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.metrics')) AS m
WHERE json_extract(r.value, '$.kind') = 'internals' AND json_type(r.value, '$.step') = 'integer';
-- dataset: internals_summary
SELECT json_extract(r.value, '$.label') AS run, json_extract(r.value, '$.internals.family') AS family,
       json_extract(r.value, '$.internals.training_run') AS training_run,
       COALESCE(CAST(json_extract(r.value, '$.step') AS TEXT), 'Not recorded') AS step,
       m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.metrics')) AS m
WHERE json_extract(r.value, '$.kind') = 'internals';
-- dataset: internals_spectrum
-- viewer.dashboard adds log10_eigenvalue; components are sorted descending.
SELECT json_extract(r.value, '$.label') AS run, json_extract(r.value, '$.internals.family') AS family,
       json_extract(r.value, '$.internals.training_run') AS training_run,
       CAST(e.key AS INTEGER) + 1 AS component, e.value AS eigenvalue
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.internals.series.spectrum')) AS e
WHERE json_extract(r.value, '$.kind') = 'internals';
-- dataset: internals_horizon
SELECT json_extract(r.value, '$.label') AS run, json_extract(r.value, '$.internals.family') AS family,
       json_extract(r.value, '$.internals.training_run') AS training_run,
       json_extract(r.value, '$.internals.series.horizon.horizon[' || v.key || ']') AS horizon,
       m.key AS metric, v.value AS value
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.internals.series.horizon')) AS m,
     json_each(m.value) AS v
WHERE json_extract(r.value, '$.kind') = 'internals' AND m.key != 'horizon';
-- dataset: internals_probe
SELECT json_extract(r.value, '$.label') AS run, json_extract(r.value, '$.internals.family') AS family,
       json_extract(r.value, '$.internals.training_run') AS training_run,
       json_extract(r.value, '$.step') AS step, p.key AS target, p.value AS r2
FROM json_each(:reconciled_runs) AS r, json_each(json_extract(r.value, '$.internals.series.probe_r2')) AS p
WHERE json_extract(r.value, '$.kind') = 'internals';
-- dataset: training_internals
-- Opt-in training-time internals rows (kind: internals) of training runs; sampled like curves.
SELECT json_extract(r.value, '$.label') AS run, json_extract(t.value, '$.step') AS step,
       m.key AS metric, m.value AS value
FROM json_each(:reconciled_runs) AS r,
     json_each(json_extract(r.value, '$.sampled_internals')) AS t, json_each(t.value) AS m
WHERE m.key NOT IN ('step', 'elapsed_seconds', 'examples');
