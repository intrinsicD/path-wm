"""Bounded lexical selection and exact evidence copies; no learned language.

python -m experiments.context_retrieval --output runs/context_s17 --seed 17
See docs/context-retrieval-plan.md for preregistered gates and limitations.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn

from pathwm.data.episode_facts import WORDS, batch, features, phrase
from pathwm.models.tasks import ContextSelector, TaskPolicy
from pathwm.world_state.context import WorkingContext
from pathwm.world_state.store import WorldStore
from pathwm.io import Run, atomic_json, digest, seed_everything, state_hash
from pathwm.evaluation.report import write_report

CONTRACTS = {f'attribute{i}': ('exact-fact', 'v1') for i in range(16)}
STRATA = ('delayed', 'distractor', 'corrected', 'resumed', 'unknown')
ARMS = ('learned', 'counted', 'canonical', 'rule', 'random', 'no_read')


class LexicalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(24, 8)
        self.policy = TaskPolicy(16, context_selector=ContextSelector(16))
        # Coarse operation/modality heads are not used or trained in this slice.
        for module in (self.policy.process, self.policy.operation,
                       self.policy.modality, self.policy.completion):
            module.requires_grad_(False)
        self.register_buffer('counts', torch.zeros(2, 24, 4, dtype=torch.long))

    def forward(self, query, keys, valid):
        return self.policy.rank_context(self.embedding(query).flatten(-2),
                                        self.embedding(keys).flatten(-2), valid)

    @torch.no_grad()
    def count_labels(self, data):
        positive = data['target'] != 32
        # Uses only selected canonical keys from the same positive labels as CE.
        keys = data['keys'][positive, data['target'][positive]]
        for slot in range(2):
            index = data['query'][positive, slot] * 4 + (keys[:, slot] % 4)
            self.counts[slot].view(-1).add_(torch.bincount(index, minlength=96))


def choose(model, query, keys, valid, arm, generator):
    if arm == 'learned':
        return model(query, keys, valid).argmax(-1)
    if arm == 'no_read':
        return torch.full((len(query),), 32, dtype=torch.long)
    if arm == 'random':
        scores = torch.rand(len(query), 33, generator=generator)
        scores[:, :32].masked_fill_(~valid, -1)
        return scores.argmax(-1)
    if arm == 'counted':
        mapped = torch.stack([model.counts[s][query[:, s]].argmax(-1) + 4*s for s in range(2)], -1)
        known = torch.stack([model.counts[s][query[:, s]].sum(-1) > 0 for s in range(2)], -1).all(-1)
    elif arm == 'rule':
        mapped, known = query % 8, torch.ones(len(query), dtype=torch.bool)
    elif arm == 'canonical':
        mapped, known = query, (query < 8).all(-1)
    else:
        raise ValueError('Unknown arm')
    match = (keys == mapped[:, None]).all(-1) & valid & known[:, None]
    return torch.where(match.any(-1), match.long().argmax(-1), 32)


def write_fact(store, event, entity, attribute, text, *, supersedes=None):
    tx = store.begin(event, occurred_at=float(store.revision), available_at=float(store.revision))
    if not any(e.id == entity for e in store.entities()):
        tx.create_entity(entity_id=entity)
    proof = tx.add_evidence('person', 'text', data={'text': text, 'entity_id': entity})
    if supersedes is not None:
        tx.supersede(supersedes, proof)
    cid = tx.put_component(entity, f'attribute{attribute}', torch.empty(0),
                           space='exact-fact', model_version='v1', evidence=(proof,),
                           role='inferred', data={'text': text})
    store.commit(tx)
    return cid, proof


def wilson(success, total):
    if total == 0:
        return [0., 1.]
    p, z = success / total, 1.96
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z * ((p*(1-p)/total + z*z/(4*total*total))**.5) / (1+z*z/total)
    return [center-radius, center+radius]


@torch.no_grad()
def evaluate(model, seed, worlds, split, directory):
    """Each world has one authoritative store; every arm sees the same event prefix.

    Metadata index is prepared during writes. Candidate scoring never reads payloads.
    Validation/copy occurs synchronously; this is not a concurrent external emitter.
    """
    gen = torch.Generator().manual_seed(seed)
    metrics, records = {}, []
    before = state_hash(model)
    keys_scanned, stale_checks, bytes_max, pin_bytes_max = 0, 0, 0, 0
    reset_checks = swap_checks = 0
    metadata_bytes_max = 0
    identity_ablation_correct = identity_ablation_count = 0
    layouts = ('flat', 'local_global')
    for world in range(worlds):
        data = batch(gen, 1, split)
        attrs, entities = data['attributes'][0], data['entities'][0]
        store, ids, proofs, values = WorldStore(), [], [], []
        entity_ids = [f'{seed}-{world}-entity-{i}' for i in range(4)]
        # Writes use exact arbitrary strings/numbers; no answer prior correlates with keys.
        for index in range(32):
            value = f' 0{int(torch.randint(100000, (1,), generator=gen))}.20 µ / "{world}"\n'
            cid, proof = write_fact(store, f'fact{index}', entity_ids[int(entities[index])],
                                    int(attrs[index]), value)
            ids.append(cid); proofs.append(proof); values.append(value)
        contexts = {(arm, layout): WorkingContext(store, capacity=4, layout=layout,
                    representations=CONTRACTS, owner=f'{seed}/{world}/{split}') for arm in ARMS for layout in layouts}
        delayed_targets = []
        for qi in range(40):
            stratum = STRATA[qi // 8]
            # Alternate A/B episodes; resumed questions recall prior A/B exact values.
            target_entity = qi % 2 if stratum == 'resumed' else int(torch.randint(4, (1,), generator=gen))
            indices = torch.where(entities == target_entity)[0]
            selected = int(indices[qi % 8])
            if stratum == 'delayed':
                selected = qi
                target_entity = int(entities[selected])
                delayed_targets.append(selected)
            elif stratum == 'resumed':
                selected = delayed_targets[qi % 8]
                target_entity = int(entities[selected])
            attribute = int(attrs[selected])
            if stratum == 'unknown':
                present = set(attrs[indices].tolist())
                near = [a for a in range(16) if a not in present and
                        any(a//4 == b//4 or a%4 == b%4 for b in present)]
                attribute = near[qi % len(near)]
            if stratum == 'corrected':
                stale = WorkingContext(store, capacity=4, representations=CONTRACTS)
                stale.retain(ids[selected], task='probe')
                values[selected] = f'corrected 00{int(torch.randint(100000, (1,), generator=gen))}.00 ß'
                ids[selected], proofs[selected] = write_fact(store, f'correction{qi}',
                    entity_ids[target_entity], attribute, values[selected], supersedes=proofs[selected])
                try:
                    stale.read('probe')
                except ValueError:
                    stale_checks += 1
                else:
                    raise AssertionError('Superseded context accepted')
                # Invalidation is deterministic. Reset affected derived views explicitly.
                for context in contexts.values():
                    context.reset()
            query = phrase(torch.tensor([attribute]), gen, split)
            query, keys, valid = features(attrs[None], entities[None], torch.tensor([target_entity]), query)
            gold_match = valid[0] & (attrs == attribute)
            gold = int(gold_match.long().argmax()) if gold_match.any() else 32
            expected = None if gold == 32 else values[gold]
            task = 'A' if target_entity % 2 == 0 else 'B'
            # Same canonical metadata, independently shuffled for every query.
            permutation = torch.randperm(32, generator=gen)
            shuffled_keys, shuffled_valid = keys[:, permutation], valid[:, permutation]
            metadata_bytes_max = max(metadata_bytes_max, len(json.dumps([
                {'id':ids[i], 'entity':entity_ids[int(entities[i])], 'attribute':int(attrs[i])}
                for i in permutation.tolist()]).encode()))
            unmasked_pick = int(choose(model, query, shuffled_keys, torch.ones_like(shuffled_valid), 'learned', gen)[0])
            unmasked_index = 32 if unmasked_pick == 32 else int(permutation[unmasked_pick])
            identity_ablation_correct += unmasked_index == gold
            identity_ablation_count += 1
            epoch = store.revision
            for arm in ARMS:
                t0 = time.perf_counter_ns()
                pick = int(choose(model, query, shuffled_keys, shuffled_valid, arm, gen)[0])
                selector_ns = time.perf_counter_ns() - t0
                index = 32 if pick == 32 else int(permutation[pick])
                # Repeat selector result across layouts to avoid random-arm confounds.
                for layout in layouts:
                    context = contexts[arm, layout]
                    start = time.perf_counter_ns()
                    answer, payload_reads, copied_bytes = None, 0, 0
                    if index != 32:
                        # One query returns one exact record. Keep the opposite episode's
                        # pins; explicitly replace this task's previous response context.
                        context.reset(task)
                        context.retain(ids[index], task=task,
                                       scope='global' if stratum == 'resumed' else 'local')
                        selected_records = context.read(task, max_components=2, max_values=0, expected_revision=epoch)
                        component = selected_records[-1]
                        answer = component.data['text']
                        payload_reads = len(selected_records)
                        copied_bytes = len(answer.encode('utf-8'))
                        assert answer.encode('utf-8') == values[index].encode('utf-8')
                    context.validate(task, expected_revision=epoch)
                    end_ns = time.perf_counter_ns() - start + selector_ns
                    saved = context.snapshot()
                    assert len(saved['entries']) <= 4 and payload_reads <= 2
                    pin_bytes_max = max(pin_bytes_max, len(json.dumps(saved).encode()))
                    if arm == 'learned' and stratum == 'resumed':
                        for intervention in ('reset', 'swap'):
                            altered = WorkingContext(store, capacity=4, layout=layout, representations=CONTRACTS)
                            if intervention == 'swap':
                                for pin in saved['entries']:
                                    altered.retain(pin['component_id'], task='B' if pin['task']=='A' else 'A', scope=pin['scope'])
                            altered.reset(task)
                            changed_answer = None
                            if index != 32:
                                altered.retain(ids[index], task=task, scope='global')
                                changed_answer = altered.read(task, max_components=2, max_values=0, expected_revision=epoch)[-1].data['text']
                            assert changed_answer == answer
                            if intervention == 'reset':
                                reset_checks += 1
                            else:
                                swap_checks += 1
                    key = f'{arm}/{layout}'
                    row = dict(world=world, query=qi, split=split, stratum=stratum, arm=arm, layout=layout,
                        correct=answer == expected, predicted_null=index == 32, gold_null=gold == 32,
                        omitted=index == 32 and gold != 32, wrong_detail=index != gold and index != 32,
                        scanned_keys=32, payload_reads=payload_reads, payload_bytes=copied_bytes,
                        retained_refs=len(saved['entries']), evictions=context.evictions,
                        selector_ns=selector_ns, end_to_end_ns=end_ns,
                        source_event_lag=None if gold==32 else store.revision-store.component(ids[gold]).revision,
                        expected=expected, answer=answer, phrase=[WORDS[t] for t in query[0].tolist()],
                        selected_id=None if index == 32 else ids[index])
                    records.append(row)
                    keys_scanned += 32
            if qi == 23:
                # Exact restart of store + all context views at a real interruption boundary.
                store = WorldStore.restore(json.loads(json.dumps(store.snapshot())))
                contexts = {k: WorkingContext.restore(store, c.snapshot(), representations=CONTRACTS, owner=f'{seed}/{world}/{split}', layout=k[1])
                            for k, c in contexts.items()}
        bytes_max = max(bytes_max, len(json.dumps(store.snapshot()).encode()))
    assert state_hash(model) == before
    for arm in ARMS:
        for layout in layouts:
            rows = [r for r in records if r['arm'] == arm and r['layout'] == layout]
            by = {s: float(np.mean([r['correct'] for r in rows if r['stratum'] == s])) for s in STRATA}
            tp = sum(r['predicted_null'] and r['gold_null'] for r in rows)
            predicted = sum(r['predicted_null'] for r in rows)
            actual = sum(r['gold_null'] for r in rows)
            precision, recall = tp / max(1, predicted), tp / max(1, actual)
            macro = float(np.mean(list(by.values())))
            metrics[f'{arm}/{layout}'] = dict(macro=macro, **by, unknown_precision=precision,
                unknown_recall=recall, unknown_precision_wilson=wilson(tp, predicted),
                unknown_recall_wilson=wilson(tp, actual), false_abstain=sum(r['omitted'] for r in rows)/max(1,len(rows)-actual),
                selector_p50_us=float(np.median([r['selector_ns'] for r in rows]))/1000,
                selector_p95_us=float(np.percentile([r['selector_ns'] for r in rows],95))/1000,
                end_to_end_p50_us=float(np.median([r['end_to_end_ns'] for r in rows]))/1000,
                end_to_end_p95_us=float(np.percentile([r['end_to_end_ns'] for r in rows],95))/1000,
                gate=macro >= .95 and min(by.values()) >= .9 and precision >= .95 and recall >= .95)
    if directory is not None:
        p = Path(directory) / f'{split}-predictions.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    examples = [next(r for r in records if r['arm']=='learned' and r['layout']=='flat' and r['stratum']==stratum)
                for stratum in STRATA]
    return dict(metrics=metrics, examples=examples, worlds=worlds, queries_per_arm=worlds*40, seed=seed,
                dataset_sha256=digest([{k:r[k] for k in ('world','query','phrase','expected')}
                                       for r in records if r['arm']=='rule' and r['layout']=='flat']),
                stale_injections_rejected=stale_checks, expected_stale_checks=worlds*8,
                exact_selected_payload_preserved=True, restart_completed=True,
                external_store_max_serialized_bytes=bytes_max, context_max_serialized_bytes=pin_bytes_max,
                keys_scanned_all_arms=keys_scanned, candidate_metadata_max_serialized_bytes=metadata_bytes_max,
                transient_tensor_bytes=2*8+32*2*8+32+33*4,
                context_reset_identical=reset_checks, context_swap_identical=swap_checks,
                expected_context_interventions=worlds*8*2,
                without_entity_mask_accuracy=identity_ablation_correct/max(1,identity_ablation_count))


@torch.no_grad()
def episode_demo(model, directory):
    """Actual one-session receive/read/interrupt/correct/emit/restart integration.

    The population experiment isolates key selection; this small separate check
    exercises the existing episode protocol, not learned conversation control.
    """
    from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
    from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
    from pathwm.models.hybrid_memory import HybridMemory
    from pathwm.world_state.modules import AssociationBinder, ReplaceUpdater, ContextEncoder
    from pathwm.world_state.session import WorldSession
    from pathwm.world_state.episodes import EpisodeClient
    agent = BeliefAgent(width=16, context_tokens=4, latent_groups=8, latent_codes=8,
        evidence_tokens=4, encoders={}, decoders={}, updater=BeliefCorrection(16,8,8,2),
        dynamics=BeliefDynamics(16,8,8,2), thinker=Thinker(16),
        memory=HybridMemory(16,recent=2,block=2,blocks=1),
        action_head=ActionHead(16), monitor=ErrorMonitor(16)).eval()
    agent.add_module('lexical_policy', model)
    binder = AssociationBinder(); binder.scorer.eval()
    modules = dict(agent=agent, binder=binder, updater=ReplaceUpdater(1).eval(),
        context_encoder=ContextEncoder(16, {n:nn.Linear(1,16) for n in CONTRACTS}, CONTRACTS).eval())
    session = WorldSession(**modules)
    client = EpisodeClient(session, representations=CONTRACTS)
    client.create('person', 'person', kind='instance')
    client.create('chat', 'chat')
    proofs, ids = [], []
    for i in range(8):
        proof = client.observe(f'proof{i}', 'person', source='person', modality='text',
            occurred_at=i+1., available_at=i+1., data={'text':f' 0{i}.20 µ '})
        proofs.append(proof)
        ids.append(client.publish(f'publish{i}', 'person', f'attribute{i}', torch.zeros(1),
            evidence=(proof,), data={'text':f' 0{i}.20 µ '}))
    client.utterance('start', 'chat', turn_id='q', phase='start', source='person', time=9., text='shipping date')
    waiting = client.response('chat', client.load('chat', ())) == 'wait'
    client.utterance('end', 'chat', turn_id='q', phase='end', source='person', time=10.)
    attributes = torch.arange(32)[None] % 16
    keys = torch.stack((attributes//4, attributes%4+4), -1)
    query = torch.tensor([[0,5]])
    valid = (torch.arange(32)[None] < 8)
    pick = int(model(query, keys, valid).argmax(-1))
    checks = dict(wait_incomplete=waiting, learned_selected_expected=pick==1)
    # Protocol checks use a valid supplied reference even in a tiny untrained smoke.
    # This separation is explicit: protocol success cannot repair selector failure.
    name = 'attribute1'
    context = WorkingContext(session, representations=CONTRACTS, owner='episode-demo')
    context.retain(ids[1], task='chat', scope='global')
    dependency = client.load('person', (name,))
    main = client.load('chat', ())
    text = context.read('chat')[-1].data['text']
    snapshot = session.snapshot()
    context_snapshot = context.snapshot()
    torch.save(snapshot, Path(directory)/'episode-session.pt')
    atomic_json(Path(directory)/'episode-context.json', context_snapshot)
    restored = WorldSession.restore(snapshot, **modules)
    other = EpisodeClient(restored, representations=CONTRACTS)
    retained = WorkingContext.restore(restored, context_snapshot,
                                     representations=CONTRACTS, owner='episode-demo')
    checks['restart_exact'] = retained.read('chat')[-1].data['text'] == text
    # A new partial turn interrupts the old completed input's proposed output.
    other.utterance('interrupt', 'chat', turn_id='next', phase='start', source='person', time=11.)
    try:
        other.emit('invalid-interrupted', 'chat', main, dependencies=(dependency,), text=text)
    except ValueError:
        checks['interrupted_rejected'] = True
    else:
        checks['interrupted_rejected'] = False
    other.utterance('resume-turn', 'chat', turn_id='next', phase='end', source='person', time=12.)
    replacement = other.observe('revised', 'person', source='person', modality='text',
        occurred_at=13., available_at=13., supersedes=proofs[1], data={'text':' corrected 0001.00 ß '})
    try:
        other.emit('invalid-stale', 'chat', other.load('chat', ()), dependencies=(dependency,), text=text)
    except ValueError:
        checks['stale_emit_rejected'] = True
    else:
        checks['stale_emit_rejected'] = False
    current_id = other.publish('rederived', 'person', name, torch.zeros(1), evidence=(replacement,),
                              data={'text':' corrected 0001.00 ß '})
    retained.reset('chat'); retained.retain(current_id, task='chat', scope='global')
    current = retained.read('chat')[-1]
    other.emit('valid-response', 'chat', other.load('chat', ()),
               dependencies=(other.load('person', (name,)),), text=current.data['text'])
    checks['current_emit'] = other.emission_status('valid-response')['status'] == 'current'
    checks['exact_revised_text'] = current.data['text'] == ' corrected 0001.00 ß '
    checks['same_authoritative_store'] = retained.store is restored.store
    checks['no_generated_source'] = not any(e.id.startswith('valid-response') for e in restored.store.evidence())
    atomic_json(Path(directory)/'episode-checks.json', checks)
    return checks


def execute(output, *, seed=17, steps=1200, worlds=64, resume=False, stop_after=None, final=False):
    seed_everything(seed)
    model = LexicalPolicy()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
    settings = dict(seed=seed, steps=steps, worlds=worlds, batch=64, precision='float32',
                    train_limit_seconds=600, final=final)
    run = Run(output, settings=settings, data=dict(generator='episode-facts-v1',
              train_seed=seed+1009, validation_seed=624925, test_seed=624927,
              pairing='both alias slots held out'), recipe=__file__, model=model,
              optimizer=optimizer, device='cpu', resume=resume)
    t0, activation_bytes, step_times = time.perf_counter(), 0, []
    try:
        while run.step < steps:
            if stop_after is not None and run.step >= stop_after:
                run.save(); run.status('paused', 'pending'); write_report(run.path)
                return model, run
            if time.perf_counter()-t0 > 600:
                raise TimeoutError('Declared training wall-time ceiling reached')
            start = time.perf_counter()
            data = batch(run.sampler, 64)
            optimizer.zero_grad(set_to_none=True)
            saved_bytes = [0]
            def pack(tensor):
                saved_bytes[0] += tensor.numel()*tensor.element_size()
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                logits = model(data['query'], data['keys'], data['valid'])
                loss = nn.functional.cross_entropy(logits, data['target'])
            loss.backward(); optimizer.step(); model.count_labels(data)
            run.step += 1
            step_times.append(time.perf_counter()-start)
            activation_bytes = max(activation_bytes, saved_bytes[0])
            if run.step == 1 or run.step % 25 == 0 or run.step == steps:
                run.log(dict(step=run.step, split='train', loss=float(loss.detach()),
                             accuracy=float((logits.argmax(-1)==data['target']).float().mean()),
                             step_seconds=step_times[-1]))
            if run.step % 100 == 0:
                run.save()
        run.save()
        model.eval()
        population = 624927 if final else 624925
        results = {s: evaluate(model, population, worlds, s, run.path) for s in (('iid','pairing') if final else ('iid',))}
        metrics = {f'{s}/{arm}': values for s, result in results.items() for arm, values in result['metrics'].items()}
        gates = [metrics[f'{s}/learned/{layout}']['gate'] for s in results for layout in ('flat','local_global')]
        invariant = all(r['stale_injections_rejected']==r['expected_stale_checks'] and
                        r['metrics']['rule/flat']['macro']==1 and r['metrics']['rule/local_global']['macro']==1
                        for r in results.values())
        episode = episode_demo(model, run.path)
        invariant = invariant and all(episode.values())
        cost = dict(total_parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    optimizer_tensor_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                                               for v in state.values() if isinstance(v,torch.Tensor)),
                    saved_for_backward_bytes=activation_bytes,
                    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    mean_step_seconds=float(np.mean(step_times)) if step_times else None,
                    current_invocation_seconds=time.perf_counter()-t0)
        result = dict(evaluation_scope='Agreement with current source assertions; supplied entity association and exact template copy',
                      gate=all(gates) and invariant, metrics=metrics, populations=results, compute=cost, episode_checks=episode,
                      limitations=['No learned language, concept induction, source credibility or entity association.',
                                   'Store invariants perform correction and stale rejection.',
                                   'Layouts share complete bounded metadata search; no retention dependency established.',
                                   'Latency includes selection and context validation/copy, excludes offline ingestion.',
                                   'Saved tensor bytes are retained autograd tensors, not peak activation allocator memory.',
                                   'Reconstruction is a separate frozen codec regression.'])
        atomic_json(run.path/'result.json', result)
        run.status('completed','pending')
        write_report(run.path)
        return model, run
    except Exception as exc:
        run.save()
        previous = json.loads((run.path/'status.json').read_text())
        run.status('completed' if previous['result']=='completed' else 'failed', 'failed', str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--final', action='store_true')
    args = vars(parser.parse_args())
    execute(**args)


if __name__ == '__main__':
    main()
