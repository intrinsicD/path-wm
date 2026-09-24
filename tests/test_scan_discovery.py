import json
import pytest
from experiments.scan_discovery import ExactIndex, make_world, query, correct, execute, CONTRACTS
from pathwm.world_state.context import WorkingContext


def test_index_discovers_beyond_scan_and_reset_preserves_exact_value():
    store, facts = make_world(924300, 32)
    index = ExactIndex(store)
    key, expected = facts[-1]['key'], facts[-1]['text']
    context = WorkingContext(store, representations=CONTRACTS)
    assert query(store, index, context, key, 'scan', 4)['status'] == 'omitted'
    for arm in ('index', 'cached', 'reset', 'full'):
        result = query(store, index, context, key, arm, 4)
        assert result['answer'] == expected
        assert result['payload_reads'] == 1
    assert query(store, index, context, ('missing', 'key'), 'index', 4)['status'] == 'unknown'


def test_correction_epoch_stale_cache_and_retraction_no_resurrection():
    store, facts = make_world(924300, 16)
    index = ExactIndex(store)
    context = WorkingContext(store, representations=CONTRACTS)
    fact = facts[-1]
    query(store, index, context, fact['key'], 'cached', 4)
    replacement = correct(store, fact, ' corrected 001.00 µ\n', 'correction')
    with pytest.raises(ValueError, match='epoch'):
        index.lookup(fact['key'])
    index.rebuild()
    result = query(store, index, context, fact['key'], 'cached', 4)
    assert result['stale_rejected'] == 1 and result['answer'] == replacement['text']
    tx = store.begin('retract', occurred_at=3., available_at=3., kind='correction')
    tx.retract_evidence(replacement['proof'])
    store.commit(tx)
    index.rebuild()
    assert query(store, index, context, fact['key'], 'cached', 4)['status'] == 'unknown'


def test_logical_restart_and_hidden_validation_work():
    store, facts = make_world(924300, 16)
    index = ExactIndex(store)
    context = WorkingContext(store, representations=CONTRACTS)
    first = query(store, index, context, facts[-1]['key'], 'cached', 4)
    restored = type(store).restore(json.loads(json.dumps(store.snapshot())))
    other = WorkingContext.restore(restored, context.snapshot(), representations=CONTRACTS)
    second = query(restored, ExactIndex(restored), other, facts[-1]['key'], 'cached', 4)
    assert second['answer'] == first['answer']
    assert second['validation_headers'] >= 16
    assert second['candidate_probes'] <= 4
    assert ExactIndex(restored).snapshot() == index.snapshot()


def test_recipe_raw_rows_and_report_failure(tmp_path, monkeypatch):
    import experiments.scan_discovery as recipe
    result = execute(tmp_path/'ok', seed=924300, records=32, queries=16)
    rows = [json.loads(s) for s in (tmp_path/'ok'/'predictions.jsonl').read_text().splitlines()]
    for arm in ('index', 'cached', 'reset'):
        selected = [r for r in rows if r['arm'] == arm]
        assert all(r['answer'] == r['expected'] for r in selected)
        assert result['metrics'][arm]['accuracy'] == 1
    assert json.loads((tmp_path/'ok'/'report.qa.json').read_text())['self_contained']
    def fail(*args, **kwargs):
        raise RuntimeError('renderer failure')
    monkeypatch.setattr(recipe, 'write_report', fail)
    with pytest.raises(RuntimeError, match='renderer failure'):
        execute(tmp_path/'broken', seed=924300, records=16, queries=16)
    assert json.loads((tmp_path/'broken'/'status.json').read_text())['result'] == 'completed'
