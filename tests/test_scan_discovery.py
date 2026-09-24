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
    with pytest.raises(ValueError, match='Stale'):
        WorkingContext.restore(store, context.snapshot(), representations=CONTRACTS)
    index.rebuild()
    result = query(store, index, context, fact['key'], 'cached', 4)
    assert result['stale_rejected'] == 1 and result['answer'] == replacement['text']
    tx = store.begin('retract', occurred_at=3., available_at=3., kind='correction', payload={'source': 'person'})
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


def test_exact_descriptor_matching_and_poisoned_index_fail_closed():
    store, facts = make_world(924300, 16)
    index = ExactIndex(store)
    context = WorkingContext(store, representations=CONTRACTS)
    key = facts[-1]['key']
    for absent in (key[::-1], (key[0]+' ',key[1]), (key[0].upper(),key[1]), ('é',key[1]), ('e\u0301',key[1])):
        assert query(store,index,context,absent,'index')['status'] == 'unknown'
    index.entries[('nonexistent','key')] = facts[0]['cid']
    with pytest.raises(ValueError, match='descriptor'):
        query(store,index,context,('nonexistent','key'),'index')
    index.entries[key] = facts[0]['cid']
    with pytest.raises(ValueError, match='descriptor'):
        query(store,index,context,key,'index')


def test_independent_meter_recount_and_stale_trace():
    store, facts = make_world(924300,16)
    index = ExactIndex(store)
    context = WorkingContext(store,representations=CONTRACTS)
    initial = query(store,index,context,facts[-1]['key'],'cached')
    assert initial['validation_headers'] == 2*16
    replacement = correct(store,facts[-1],'next','revision')
    index.rebuild()
    fresh = query(store,index,context,replacement['key'],'cached')
    assert fresh['validation_headers'] == 3*17
    assert fresh['trace'] == ['cache_stale_rejected','index_lookup','source_validated']
    assert fresh['candidate_probes'] == 2
    assert fresh['materialized_bytes'] > fresh['payload_bytes']


def test_replay_matches_uninterrupted_future_and_permuted_enumeration(monkeypatch):
    store, facts = make_world(924300,16)
    uninterrupted_index = ExactIndex(store)
    a = WorkingContext(store,representations=CONTRACTS)
    query(store,uninterrupted_index,a,facts[-1]['key'],'cached')
    resumed_store = type(store).restore(json.loads(json.dumps(store.snapshot())))
    b = WorkingContext.restore(resumed_store,a.snapshot(),representations=CONTRACTS)
    resumed_index = ExactIndex(resumed_store)
    original = resumed_store.components
    monkeypatch.setattr(resumed_store,'components',lambda *args,**kwargs: tuple(reversed(original(*args,**kwargs))))
    resumed_index.rebuild()
    assert resumed_index.snapshot() == uninterrupted_index.snapshot()
    for step in range(8):
        for source in (store,resumed_store):
            replacement = correct(source,facts[-1],f'future {step}',f'future{step}')
        facts[-1] = replacement
        uninterrupted_index.rebuild(); resumed_index.rebuild()
        for fact in facts:
            x=query(store,uninterrupted_index,a,fact['key'],'cached')
            y=query(resumed_store,resumed_index,b,fact['key'],'cached')
            assert {k:v for k,v in x.items() if k!='query_ns'} == {k:v for k,v in y.items() if k!='query_ns'}
        assert a.snapshot() == b.snapshot()
        assert store.snapshot() == resumed_store.snapshot()


def test_duplicate_keys_refuse_and_skipped_update_does_not_look_absent():
    store,facts=make_world(924300,16)
    index=ExactIndex(store)
    context=WorkingContext(store,representations=CONTRACTS)
    correct(store,facts[-1],'changed','change')
    with pytest.raises(ValueError,match='epoch'):
        query(store,index,context,facts[-1]['key'],'index')
    # Duplicate active descriptors across distinct slots are rejected globally;
    # this narrow recipe intentionally has no conflict-resolving query policy.
    duplicate=dict(facts[0],key=facts[1]['key'])
    correct(store,duplicate,'duplicate','duplicate')
    with pytest.raises(ValueError,match='Ambiguous'):
        index.rebuild()
