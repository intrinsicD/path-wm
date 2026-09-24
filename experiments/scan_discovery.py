"""Exact-descriptor indexed feasibility screen; no learned retrieval claim.

python -m experiments.scan_discovery --output runs/my_discovery --seed 924301
See docs/scan-discovery-plan.md for fixed gates, scope and accounting limits.
"""
import argparse
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from pathwm.evaluation.report import write_report
from pathwm.io import atomic_json, digest, environment, source_record
from pathwm.world_state.context import WorkingContext
from pathwm.world_state.records import Limits
from pathwm.world_state.store import WorldStore

CONTRACTS = {f'detail{i}': ('exact-descriptor', 'v1') for i in range(4)}
ARMS = ('index', 'cached', 'reset', 'scan', 'full')


class MeteredStore(WorldStore):
    """Instrumentation only: the inherited store remains authoritative.

    Header visits count the actual linear filter in WorldStore.components.
    Materialization bytes count serialized returned records, not allocator traffic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers = self.materialized = 0

    def components(self, entity_id=None, name=None):
        self.headers += len(self._components)
        result = super().components(entity_id, name)
        self.materialized += sum(len(json.dumps(vars(c), ensure_ascii=False).encode()) for c in result)
        return result

    def component(self, key):
        result = super().component(key)
        self.materialized += len(json.dumps(vars(result), ensure_ascii=False).encode())
        return result


class ExactIndex:
    """Recipe-local complete index for exact ordered descriptor pairs.

    Rebuild after every source mutation, including retractions. A stale epoch is
    an error, never a certified unknown. No index is a second source of truth.
    """
    def __init__(self, store):
        self.store = store
        self.rebuild()

    def rebuild(self):
        heads = {}
        for c in self.store.components():
            slot = (c.entity_id, c.name)
            old = heads.get(slot)
            if old is None or (c.valid_from, c.revision, c.id) > (old.valid_from, old.revision, old.id):
                heads[slot] = c
        entries = {}
        for c in heads.values():
            if not c.active:
                continue
            key = tuple(c.data['key'])
            if key in entries:
                raise ValueError('Ambiguous exact descriptor')
            entries[key] = c.id
        self.entries, self.revision = entries, self.store.revision

    def lookup(self, key):
        if self.revision != self.store.revision:
            raise ValueError('Stale index epoch')
        return self.entries.get(tuple(key))

    def snapshot(self):
        return {'revision': self.revision, 'entries': sorted([list(k), v] for k, v in self.entries.items())}


def make_world(seed, records=256):
    if type(records) is not int or not 16 <= records <= 384 or records % 4:
        raise ValueError('records must be a multiple of four in [16,384]')
    rng = random.Random(seed)
    store = MeteredStore(Limits(entities=128, components=2048))
    facts = []
    # Shared-token distractors, unique ordered pairs; never query a component ID.
    pairs = [(f'{seed}-a{i}', f'{seed}-b{j}') for i in range(32) for j in range(32)]
    rng.shuffle(pairs)
    assert len(set(pairs[:records])) == records
    tx = store.begin(f'{seed}-initial', occurred_at=0., available_at=0.)
    for i in range(records):
        entity, name = f'{seed}-entity{i//4}', f'detail{i%4}'
        if i % 4 == 0:
            tx.create_entity(entity_id=entity)
        value = f' 00{rng.randrange(10**9)}.20 µ / "{seed}:{i}"\n'
        proof = tx.add_evidence('person', 'text', data={'text': value})
        cid = tx.put_component(entity, name, torch.empty(0), space='exact-descriptor',
            model_version='v1', evidence=(proof,), role='inferred', data={'text': value, 'key': pairs[i]})
        facts.append(dict(key=pairs[i], text=value, entity=entity, name=name, cid=cid, proof=proof))
    store.commit(tx)
    return store, facts


def correct(store, fact, text, event):
    tx = store.begin(event, occurred_at=float(store.revision), available_at=float(store.revision))
    proof = tx.add_evidence('person', 'text', data={'text': text})
    tx.supersede(fact['proof'], proof)
    cid = tx.put_component(fact['entity'], fact['name'], torch.empty(0),
        space='exact-descriptor', model_version='v1', evidence=(proof,), role='inferred',
        data={'text': text, 'key': fact['key']})
    store.commit(tx)
    return dict(fact, text=text, proof=proof, cid=cid)


def query(store, index, context, key, arm, budget=16):
    if arm not in ARMS or type(budget) is not int or budget < 1:
        raise ValueError('Invalid arm or candidate budget')
    start = time.perf_counter_ns()
    h0, b0 = store.headers, store.materialized
    epoch = store.revision
    task = digest(list(key))
    probes = stale = 0
    trace, selected = [], ()
    if arm != 'cached':
        context.reset()
    else:
        try:
            selected = context.read(task, max_components=1, expected_revision=epoch)
            if selected:
                probes += 1
                trace.append('cache_valid')
        except ValueError:
            stale = 1
            probes += 1
            trace.append('cache_stale_rejected')
            context.reset(task)
    omitted = False
    if not selected:
        if arm in ('index', 'cached', 'reset'):
            trace.append('index_lookup')
            probes += 1
            cid = index.lookup(key)
        else:
            # Deliberately complete metadata preparation is charged, even for
            # the weak first-K control. This is not a physical I/O-bound scan.
            heads = {}
            for c in store.components():
                slot = (c.entity_id, c.name)
                old = heads.get(slot)
                if old is None or (c.valid_from, c.revision, c.id) > (old.valid_from, old.revision, old.id):
                    heads[slot] = c
            current = [(tuple(c.data['key']), c.id) for c in heads.values() if c.active]
            limit = len(current) if arm == 'full' else budget
            cid = None
            for candidate_key, candidate in current[:limit]:
                probes += 1
                if candidate_key == tuple(key):
                    cid = candidate
                    break
            omitted = cid is None and len(current) > limit
        if cid is not None:
            context.reset(task)
            context.retain(cid, task=task)
            selected = context.read(task, max_components=1, expected_revision=epoch)
            trace.append('source_validated')
    elif arm == 'cached':
        context.retain(selected[0].id, task=task)
    if selected and tuple(selected[0].data['key']) != tuple(key):
        raise ValueError('Retrieved descriptor does not match query')
    answer = selected[0].data['text'] if selected else None
    result = dict(answer=answer, source_id=selected[0].id if selected else None,
        status='answered' if selected else ('omitted' if omitted else 'unknown'),
        stale_rejected=stale, candidate_probes=probes, payload_reads=len(selected),
        payload_bytes=len(answer.encode()) if answer is not None else 0,
        pins=len(context.snapshot()['entries']), trace=trace,
        validation_headers=store.headers-h0, materialized_bytes=store.materialized-b0,
        historical_headers=len(store._components))
    result['query_ns'] = time.perf_counter_ns()-start
    return result


def execute(output, *, seed=924301, records=256, queries=128):
    if type(queries) is not int or queries < 16 or queries % 16:
        raise ValueError('queries must be a positive multiple of 16')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, cpu_start = time.perf_counter(), time.process_time()
    settings = dict(seed=seed, records=records, queries=queries, candidate_budget=16,
                    cache_capacity=4, purpose='indexed feasibility; no training')
    source = source_record(__file__, torch.nn.Identity())
    atomic_json(output/'run.json', dict(identity=dict(settings=settings, environment=environment('cpu')), source=source))
    (output/'recipe.py').write_text(Path(__file__).read_text())
    for i, filename in enumerate(source['files']):
        dest = output/'source'/f'{i:03d}_{Path(filename).name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(Path(filename).read_bytes())
    (output/'metrics.jsonl').write_text('')
    atomic_json(output/'status.json', dict(result='running', report='pending', step=0))
    rows, restarts, corrections = [], [], []
    try:
        t = time.perf_counter()
        store, facts = make_world(seed, records)
        # Append additional distractors in one causal transaction.
        tx = store.begin(f'{seed}-distractors', occurred_at=1., available_at=1.)
        for i in range(64):
            entity = f'{seed}-distractor{i//4}'
            if i % 4 == 0:
                tx.create_entity(entity_id=entity)
            proof = tx.add_evidence('person', 'text', data={'text': f'distractor {seed}:{i}'})
            tx.put_component(entity, f'detail{i%4}', torch.empty(0), space='exact-descriptor',
                model_version='v1', evidence=(proof,), role='inferred',
                data={'text': f'distractor {seed}:{i}', 'key': [facts[i % records]['key'][0], f'{seed}-d{i}']})
        store.commit(tx)
        ingestion_seconds = time.perf_counter()-t
        t = time.perf_counter()
        index = ExactIndex(store)
        indexing_seconds = time.perf_counter()-t
        contexts = {a: WorkingContext(store, capacity=4, representations=CONTRACTS) for a in ARMS}
        rng = random.Random(seed+1)
        recovery_seconds = correction_seconds = fixture_seconds = 0.
        prior_headers = prior_materialized = 0
        for qi in range(queries):
            if time.perf_counter()-start > 120:
                raise TimeoutError('Per-population wall budget exceeded')
            slot = qi % 16
            stratum = 'delayed' if slot < 8 else 'corrected' if slot < 12 else 'restart' if slot < 14 else 'unknown'
            # Repeat the tail key across each cycle, and repeatedly correct it:
            # exposes cache invalidation and a 32-version chain in default runs.
            target = records-1 if slot in (0, 1, 8, 9, 10, 11, 12, 13) else rng.randrange(16, records) if records > 16 else records-1
            fact = facts[target]
            if stratum == 'corrected':
                # Ensure a real old retained ref exists before the incoming correction.
                fixture_start = time.perf_counter()
                for context in contexts.values():
                    context.reset()
                    context.retain(fact['cid'], task=digest(list(fact['key'])))
                fixture_seconds += time.perf_counter()-fixture_start
                t = time.perf_counter()
                fact = facts[target] = correct(store, fact, f' corrected 000{seed}:{qi}.50 ß\n', f'{seed}-correction{qi}')
                correction_seconds += time.perf_counter()-t
                t = time.perf_counter()
                index.rebuild()
                indexing_seconds += time.perf_counter()-t
                corrections.append(dict(query=qi, source_id=fact['cid'], historical_headers=len(store._components)))
            key = (f'{seed}-missing{qi}', fact['key'][1]) if stratum == 'unknown' else fact['key']
            expected = None if stratum == 'unknown' else fact['text']
            if stratum == 'restart':
                t = time.perf_counter()
                before = {a: query(store, index, c, key, a) for a, c in contexts.items()}
                snapshot = dict(store=store.snapshot(), contexts={a: c.snapshot() for a,c in contexts.items()}, index=index.snapshot())
                atomic_json(output/f'restart-{qi}.json', snapshot)
                saved = json.loads((output/f'restart-{qi}.json').read_text())
                prior_headers += store.headers
                prior_materialized += store.materialized
                store = MeteredStore.restore(saved['store'])
                index = ExactIndex(store)
                contexts = {a: WorkingContext.restore(store, s, representations=CONTRACTS) for a,s in saved['contexts'].items()}
                cache_equal = all(contexts[a].snapshot()==saved['contexts'][a] for a in ARMS)
                after = {a: query(store, index, c, key, a) for a,c in contexts.items()}
                restarts.append(dict(query=qi, cache_equal=cache_equal, index_equal=index.snapshot()==saved['index'],
                    answers_equal=all(before[a]['answer']==after[a]['answer'] and before[a]['status']==after[a]['status'] for a in ARMS)))
                recovery_seconds += time.perf_counter()-t
            # Rotate execution order so timing does not always favor one arm.
            order = list(ARMS); rng.shuffle(order)
            for arm in order:
                row = query(store, index, contexts[arm], key, arm)
                rows.append(dict(row, query=qi, arm=arm, stratum=stratum, key=key, expected=expected,
                                 expected_source=None if expected is None else fact['cid'],
                                 expected_status='unknown' if expected is None else 'answered', revision=store.revision))
        (output/'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        atomic_json(output/'population.json', dict(seed=seed, descriptor_keys=[f['key'] for f in facts],
            final_payloads=[f['text'] for f in facts], corrections=corrections))
        atomic_json(output/'last.json', dict(store=store.snapshot(), index=index.snapshot(), contexts={a:c.snapshot() for a,c in contexts.items()}))
        metrics = {}
        for arm in ARMS:
            chosen = [r for r in rows if r['arm']==arm]
            metrics[arm] = dict(accuracy=sum(r['answer']==r['expected'] and r['status']==r['expected_status'] for r in chosen)/len(chosen),
                source_agreement=sum(r['source_id']==r['expected_source'] for r in chosen)/len(chosen),
                p95_ms=float(np.percentile([r['query_ns']/1e6 for r in chosen],95)),
                max_ms=max(r['query_ns']/1e6 for r in chosen),
                query_seconds=sum(r['query_ns'] for r in chosen)/1e9,
                max_probes=max(r['candidate_probes'] for r in chosen),
                max_headers=max(r['validation_headers'] for r in chosen),
                max_header_ratio=max(r['validation_headers']/r['historical_headers'] for r in chosen),
                materialized_bytes=sum(r['materialized_bytes'] for r in chosen),
                stale_rejections=sum(r['stale_rejected'] for r in chosen),
                omitted=sum(r['status']=='omitted' for r in chosen),
                strata={s:dict(count=len(rs := [r for r in chosen if r['stratum']==s]),
                    accuracy=sum(r['answer']==r['expected'] and r['status']==r['expected_status'] for r in rs)/len(rs),
                    p95_ms=float(np.percentile([r['query_ns']/1e6 for r in rs],95))) for s in ('delayed','corrected','restart','unknown')})
        compute = dict(ingestion_seconds=ingestion_seconds, index_build_and_rebuild_seconds=indexing_seconds,
            correction_seconds=correction_seconds, logical_restart_seconds=recovery_seconds,
            fixture_seconds=fixture_seconds, total_header_visits=prior_headers+store.headers,
            total_materialized_bytes=prior_materialized+store.materialized,
            total_wall_seconds=time.perf_counter()-start, total_cpu_seconds=time.process_time()-cpu_start,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            index_bytes=len(json.dumps(index.snapshot()).encode()), store_bytes=len(json.dumps(store.snapshot()).encode()),
            cache_bytes={a:len(json.dumps(c.snapshot()).encode()) for a,c in contexts.items()},
            training_updates=0)
        bounded = [r for r in rows if r['arm'] in ('index','cached','reset')]
        gates = dict(exact=all(r['answer']==r['expected'] and r['source_id']==r['expected_source'] and r['status']==r['expected_status'] for r in bounded),
            restart=all(r['index_equal'] and r['answers_equal'] and r['cache_equal'] for r in restarts),
            budgets=all(r['candidate_probes']<=16 and r['payload_reads']<=1 and r['pins']<=4 and
                        r['validation_headers']<=4*r['historical_headers'] for r in bounded),
            latency=all(metrics[a]['p95_ms']<=20 for a in ('index','cached','reset')),
            resources=compute['total_wall_seconds']<=120 and compute['peak_process_rss_kib']<=2*1024**2 and compute['index_bytes']<=1024**2,
            correction=metrics['cached']['stale_rejections']==queries//4)
        result = dict(evaluation_scope='Exact supplied descriptors, current source copies; candidate budget is not a total storage-work bound.',
            gate=all(gates.values()), gates=gates, metrics=metrics, compute=compute, restarts=restarts,
            population_sha256=digest(json.loads((output/'population.json').read_text())),
            examples=[next(r for r in rows if r['arm']==a and r['stratum']==s) for a in ARMS for s in ('delayed','corrected','restart','unknown')],
            limitations=['No learned model, semantic retrieval, identity inference or reconstruction/prediction result.',
                'Full scan remains physically feasible; first-16 ranking cap is an experimental constraint.',
                'Latest-head validation scans history; payload_reads counts returned payloads, not internal copies.',
                'Logical restart only; no process-crash atomicity. Timing includes instrumentation.',
                'Evaluation cost includes all arms and fixtures, excludes report rendering; not an isolated deployment benchmark.',
                'Structural report verification with unchanged renderer; no new browser QA.'])
        atomic_json(output/'result.json',result)
        (output/'metrics.jsonl').write_text(''.join(json.dumps(dict(step=0, split=a, accuracy=m['accuracy'], p95_ms=m['p95_ms']))+'\n' for a,m in metrics.items()))
        atomic_json(output/'status.json',dict(result='completed',report='pending',step=0))
        write_report(output)
        return result
    except Exception as exc:
        previous = json.loads((output/'status.json').read_text())
        atomic_json(output/'status.json',dict(result=previous['result'] if previous['result']=='completed' else 'failed', report='failed',step=0,error=str(exc)))
        if rows and not (output/'predictions.jsonl').exists():
            (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        raise
    finally:
        atomic_json(output/'invocation.json', dict(wall_seconds=time.perf_counter()-start,
                    cpu_seconds=time.process_time()-cpu_start, includes_report=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=924301)
    parser.add_argument('--records', type=int, default=256)
    parser.add_argument('--queries', type=int, default=128)
    result = execute(**vars(parser.parse_args()))
    print(json.dumps(dict(gate=result['gate'], gates=result['gates'], compute=result['compute'])))


if __name__ == '__main__':
    main()
