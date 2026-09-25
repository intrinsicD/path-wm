"""R2 software slice: a persistent-machine RuleWorld life through one composed session.

One `WorldSession` (store, clock, instance identity, belief with the shared image
encoder) and one `ConceptMemory` client (concept membership and codes). The same two
machines (fixed textures and hidden rules) persist across scenes with new layouts.
The life exercises repeated observations, demonstrated acquisition, structured-goal
planning with read-set checks, receipts, external verification, a free-text request
that must ask, and a restart from saved files.

SOFTWARE CHECK ONLY: weights are random (seeded), so identity decisions, concepts
and plans carry no capability claim. R1 C1/C2 must pass before any learned R2 run.

    .venv/bin/python -m experiments.unified_session --output runs/<dir> [--scenes 4]
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.multimodal import build_model
from pathwm.data import rule_world as rw
from pathwm.evaluation.report import write_report
from pathwm.io import atomic_json, atomic_torch, environment, file_hash, load_component, source_record, state_hash
from pathwm.models.latent_core import LatentCore
from pathwm.models.slots import SlotPerception
from pathwm.models.tasks import Actor, TaskRequest
from pathwm.world_state.concepts import AgentSettings, ConceptMemory
from pathwm.world_state.modules import (
    AssociationBinder,
    Candidate,
    CandidateEncoder,
    ContextEncoder,
    RecurrentUpdater,
)
from pathwm.world_state.session import WorldSession
from pathwm.world_state.unified import (
    ActionEncoder,
    GoalSpec,
    UnifiedAgent,
    VerificationRecord,
    model_versions,
)

WIDTH, KEY, VALUE, STATE = 64, 32, 16, 16


def describe(perception_run=None, identity_run=None, learned_keys=False, *, calibrated=False):
    """Scope, binding and limitations of THIS run's actual composition (no overclaim)."""
    if identity_run is not None:
        loaded = "perception and exported identity key from one S1 --identity run (named in run.json/models.json)"
        binding = "exported S1 identity key (uncalibrated binder thresholds)"
        keys = ("Session identity uses the exported S1 key (= core.key_head) on slot tokens with untrained, "
                "uncalibrated binder thresholds: identity decisions are a software exercise, not learned tracking.")
        if calibrated:
            binding = "exported S1 identity key with checkpoint-bound TRAIN-calibrated binder thresholds"
            keys = ("Session identity uses the exported S1 key (= core.key_head) on slot tokens with "
                    "checkpoint-bound TRAIN-calibrated thresholds. This life checks software contracts; "
                    "identity quality requires the separate native visual-memory evaluation.")
    elif learned_keys:
        loaded = ("an R1 perception checkpoint (named in run.json)" if perception_run is not None
                  else "no checkpoint")
        binding, keys = "learned slot keys (untrained)", "Session identity uses an untrained candidate key projection."
    else:
        loaded = ("an R1 perception checkpoint (named in run.json)" if perception_run is not None
                  else "no checkpoint")
        binding = "pixel-histogram fixture"
        keys = "Session identity uses the non-learned pixel-histogram SOFTWARE fixture."
    scope = (f"SOFTWARE check: loaded {loaded}; core, belief, action encoder and candidate values are random "
             "(seeded). Contracts only, no capability claim.")
    limitations = [
        f"Loaded weights: {loaded}. No loaded perception is a qualified formal configuration here.",
        "Core, belief, action encoder and candidate values are random: concept, prediction and plan quality "
        "are meaningless; no rule-application or learned-core claim.",
        keys,
        "Verifier grounding uses the pixel the agent reported for each entity.",
        "No natural data, no cross-modal binding, no learned R2 claim.",
    ]
    return dict(scope=scope, binding=binding, limitations=limitations)


KEY_ARCHITECTURE = f"pathwm.models.latent_core.key_head({WIDTH}, {KEY})"


def build(seed, perception_run=None, identity_run=None, *, shared_key=False, candidate_scope="all"):
    """Fresh modules. The belief's image encoder IS the slot consumer's encoder.

    `perception_run`: an R1 perception run directory (read-only) whose `last.pt`
    initialises the slot perception; everything else stays randomly initialised.
    `identity_run`: an S1 run trained with `--identity`; its perception AND exported
    `key` load together; the key becomes `core.key_head` and the session's candidate
    key (one module: candidate keys equal `core.key` on slot tokens). Values keep their
    own projection. Binder thresholds stay uncalibrated; no learned-tracking claim.
    `shared_key`: rebuild that composition for saved weights (see `load_models`).
    """
    if perception_run is not None and identity_run is not None:
        raise ValueError("Load perception from one run: use identity_run alone")
    if candidate_scope not in ("all", "predicted-machine"):
        raise ValueError("Unknown candidate scope")
    if candidate_scope == "predicted-machine" and identity_run is None and not shared_key:
        raise ValueError("Machine candidate scope requires the exported machine identity key")
    torch.manual_seed(seed)
    perception = SlotPerception(WIDTH, 7, 3)
    core = LatentCore(WIDTH)
    source = dict(source="own_projection")
    if perception_run is not None:
        load_component(perception, Path(perception_run) / "last.pt", "perception")
    if identity_run is not None:
        run = Path(identity_run)
        settings = json.loads((run / "run.json").read_text())["identity"]["settings"]
        if settings.get("identity") not in ("joint", "detached"):
            raise ValueError("identity_run must be an S1 run trained with --identity joint|detached")
        sizes = settings.get("sizes", {})
        if (sizes.get("width"), sizes.get("key_width")) != (WIDTH, KEY):
            raise ValueError(f"identity key architecture differs from {KEY_ARCHITECTURE}")
        load_component(perception, run / "last.pt", "perception")
        load_component(core.key_head, run / "last.pt", "key")
        source = dict(source="identity_run", run=str(run), checkpoint=str(run / "last.pt"),
                      checkpoint_sha256=file_hash(run / "last.pt"), identity_mode=settings["identity"],
                      architecture=KEY_ARCHITECTURE, key_state_sha256=state_hash(core.key_head),
                      perception_state_sha256=state_hash(perception))
        shared_key = True
    agent = build_model(width=WIDTH, state_model="belief", memory_recent=2, memory_block=2, memory_blocks=1)
    agent.encoders["image"] = perception.encoder
    candidates = CandidateEncoder(WIDTH, KEY, VALUE, key=core.key_head if shared_key else None)
    if candidate_scope != "all":
        source["candidate_scope"] = candidate_scope
    candidates.key_source = source
    actions = ActionEncoder(WIDTH, agent.action_width)
    updater = RecurrentUpdater(VALUE, STATE)
    context = ContextEncoder(WIDTH, {"state": nn.Linear(STATE, WIDTH)}, {"state": ("belief", "state-v1")})
    modules = dict(perception=perception, core=core, agent=agent, candidates=candidates, actions=actions,
                   updater=updater, context=context)
    for module in modules.values():
        module.eval()
    return modules


def save_models(modules, path):
    """Weights plus `models.json` naming the candidate-key composition and its source."""
    path = Path(path)
    atomic_torch(path, {name: m.state_dict() for name, m in modules.items()})
    atomic_json(path.with_name("models.json"), dict(candidate_key=modules["candidates"].key_source))


def load_models(path, seed=0):
    """Fresh module objects with saved weights, rebuilt in the saved composition (the
    shared image encoder and a shared key load twice, equally)."""
    path = Path(path)
    config = path.with_name("models.json")
    source = json.loads(config.read_text())["candidate_key"] if config.exists() else dict(source="own_projection")
    modules = build(seed, shared_key=source["source"] != "own_projection",
                    candidate_scope=source.get("candidate_scope", "all"))
    saved = torch.load(path, weights_only=True)
    for name, module in modules.items():
        module.load_state_dict(saved[name])
    if source["source"] != "own_projection" and state_hash(modules["core"].key_head) != source["key_state_sha256"]:
        raise ValueError("Saved candidate key differs from its recorded source")
    modules["candidates"].key_source = source
    return modules


def session_modules(modules, binder=None):
    binder = binder or AssociationBinder()
    binder.scorer.eval()
    return dict(agent=modules["agent"], binder=binder, updater=modules["updater"],
                context_encoder=modules["context"])


def versions(modules):
    return model_versions(modules["perception"], modules["core"], modules["candidates"], modules["actions"])


def new_agent(modules, directory, *, binder=None, settings=None, contract=None):
    session = WorldSession(**session_modules(modules, binder))
    memory = ConceptMemory(directory, versions=versions(modules), session=session)
    return UnifiedAgent(modules["perception"], modules["core"], session, memory, modules["candidates"],
                        modules["actions"], contract or rw.TaskContract(), settings or AgentSettings())


def restore_agent(modules, directory, *, binder=None, settings=None, contract=None):
    directory = Path(directory)
    session = WorldSession.restore(torch.load(directory / "session.pt", weights_only=True),
                                   **session_modules(modules, binder))
    memory = ConceptMemory.attach(directory, session, versions=versions(modules))
    agent = UnifiedAgent(modules["perception"], modules["core"], session, memory, modules["candidates"],
                         modules["actions"], contract or rw.TaskContract(), settings or AgentSettings())
    agent.recover()  # derivations an interrupted process may not have completed
    return agent


def pixel_candidates(agent):
    """SOFTWARE binding fixture: non-learned colour-histogram keys (4 levels per
    channel) of the pixels each slot owns under the agent's OWN alpha masks; values
    from the learned candidate encoder. No hidden IDs or masks reach binding. It
    stands in for the untrained identity scorer (an R2 learned connection)."""

    def make(percept, rgb):
        owner = percept.alpha[0].argmax(0)
        q = (rgb.clamp(0, 1) * 3.999).long()
        code = q[0] * 16 + q[1] * 4 + q[2]
        _, values = agent.candidate_encoder(percept.slots[0])
        out = []
        for k in range(len(values)):
            mask = owner == k
            if not mask.any():
                continue
            key = F.normalize(torch.bincount(code[mask], minlength=64).float().sqrt(), dim=0)
            out.append(Candidate(f"slot-{k}", "rule_world_camera", "image", key, values[k].detach(),
                                 "slot-pixel-histogram", "fixture-v1", exclusive_group="frame"))
        return tuple(out)

    return make


class RuleWorldVerifier:
    """Caller-owned verifier. The caller grounds goal entities to world machines by
    the pixel it was shown for each entity; ungrounded targets verify as unknown."""

    verifier_id = "rule_world_verifier"

    def __init__(self, world, sides):
        self.world, self.sides = world, dict(sides)

    def verify(self, goal, observed_at):
        sides = [self.sides.get(ref) for ref, _ in goal.targets]
        if None in sides:
            status = "unknown"
        else:
            ok = all(int(self.world.lamps[s]) == v for s, (_, v) in zip(sides, goal.targets))
            status = "success" if ok else "failure"
        return VerificationRecord(goal.task_id, goal.digest(), status, self.verifier_id, observed_at)


class CountingActuator(rw.Actuator):
    def __init__(self, world):
        super().__init__(world)
        self.presses = 0

    def press(self, record):
        self.presses += 1
        return super().press(record)


def demonstrations(scene, rules, lamps, generator, per_machine):
    """Teacher presses on a scratch copy of the scene: (pre, record, status, post)."""
    world = rw.RuleWorld(scene, rules, lamps, rw.TaskContract(budget=10**9))
    out = []
    for side in (0, 1):
        for _ in range(per_machine):
            world.lamps = torch.randint(2, (2,), generator=generator)
            i, j = torch.randperm(4, generator=generator)[:2].tolist()
            record = rw.ActionRecord(tuple(scene.machine_xy[0, side].tolist()),
                                     tuple(scene.object_xy[0, i].tolist()), tuple(scene.object_xy[0, j].tolist()))
            pre = world.frame()
            receipt = world.press(record)
            out.append((pre, record, receipt.status, world.frame()))
    return out


def grounding(agent, world):
    """Caller-side: which world machine each visible entity's pixel lands on."""
    sides = {}
    for m in agent.view.machines:
        hit = world.resolve(m["xy"])
        if m["instance"] is not None and hit in (1, 2):
            sides[m["instance"]] = hit - 1
    return sides


def summary(agent):
    view = agent.memory.view()
    entities = list(view["entities"].values())
    decisions = [d for e in agent.session.store.events() if e.kind == "observation"
                 for d in agent.session.decision(e.id)["bindings"]]
    transitions = [e for e in view["evidence"].values() if e.modality == "transition"]
    own = [e for e in transitions if e.data.get("perceived_in") != e.event_id]  # the agent's presses
    return dict(
        revision=agent.session.store.revision, time=agent.session.time,
        instances=sum(e.kind == "instance" for e in entities), concepts=sum(e.kind == "concept" for e in entities),
        decisions={s: sum(d["status"] == s for d in decisions) for s in ("matched", "new", "unresolved")},
        transitions=len(transitions), agent_presses=len(own),
        agent_ok_presses=sum(e.data["receipt"] == "ok" for e in own),
        attributed=sum(c.name == "attribution" and c.active for c in view["components"].values()),
        verifications=sum(e.modality == "verification" for e in view["evidence"].values()),
    )


@torch.no_grad()
def life(directory, *, seed=0, scenes=4, per_machine=4, restart_after=2, perception_run=None, fixture=True,
         identity_run=None, binding_calibration=None):
    """Persistent machines across `scenes` layouts; restart after `restart_after`.

    fixture=True binds with `pixel_candidates` (SOFTWARE); False uses the learned
    (here untrained) slot candidate encoder."""
    directory = Path(directory)
    g = torch.Generator().manual_seed(seed)
    binder, policy = visual_binding_policy(binding_calibration, identity_run)
    modules = build(seed, perception_run, identity_run, candidate_scope=policy["candidate_scope"] if policy else "all")
    save_models(modules, directory / "models.pt")
    memory_dir = directory / "memory"
    agent = new_agent(modules, memory_dir, binder=binder)
    keys = pixel_candidates(agent) if fixture else None
    kinds = torch.tensor([list(rw.KIND_SPLIT["train"][:2])])
    train = rw.split_rules()["train"]
    rules = tuple(train[int(i)] for i in torch.randperm(len(train), generator=g)[:2])
    lamps = torch.zeros(2, dtype=torch.long)
    rows, restart = [], None
    for s in range(scenes):
        scene = rw.sample_scenes(g, kinds)
        world = rw.RuleWorld(scene, rules, lamps, rw.TaskContract())
        actuator = CountingActuator(world)
        if s == 0:
            demos = demonstrations(scene, rules, lamps, g, per_machine)
            agent.observe(world.frame(), transitions=demos, candidates=keys)
            rows.append(dict(split="life", scene=s, kind="demonstration", demos=len(demos), **summary(agent)))
            continue
        agent.observe(world.frame(), candidates=keys)
        free = agent.run_task(actuator, None, request=TaskRequest(
            f"task-{s}-text", "turn-the-left-lamp-on", Actor("user", "life-harness")))
        rows.append(dict(split="life", scene=s, kind="free_text", option=free["option"],
                         actuator_presses=actuator.presses, **summary(agent)))
        targets = tuple((m["instance"], int(torch.randint(2, (1,), generator=g)))
                        for m in agent.view.machines if m["instance"] is not None)
        goal = None
        if not targets:  # nothing identified: no goal can be stated about entities
            rows.append(dict(split="life", scene=s, kind="goal", option="no_identified_machine", presses=0,
                             outcome="unknown", steps=[], **summary(agent)))
        else:
            goal = GoalSpec(f"task-{s}", "lamp_state", targets)
            verifier = RuleWorldVerifier(world, grounding(agent, world))
            result = agent.run_task(actuator, goal, verifier=verifier, candidates=keys)
            lamps = world.lamps.clone()
            rows.append(dict(split="life", scene=s, kind="goal", option=result["option"],
                             presses=result["presses"], outcome=result["outcome"], steps=result["trace"],
                             **summary(agent)))
        if s == restart_after:
            agent.save(memory_dir)
            before = dict(summary=summary(agent), plan=_plan_snapshot(agent, goal))
            modules = load_models(directory / "models.pt", seed + 1)  # fresh objects, saved weights
            agent = restore_agent(modules, memory_dir, binder=binder)
            keys = pixel_candidates(agent) if fixture else None
            agent.resume_view()
            after = dict(summary=summary(agent), plan=_plan_snapshot(agent, goal))
            restart = dict(scene=s, equal=before == after, before=before, after=after)
            rows.append(dict(split="life", scene=s, kind="restart", equal=restart["equal"], **summary(agent)))
    return rows, restart, summary(agent)


def _plan_snapshot(agent, goal):
    dispatch = agent.dispatch(None, goal)  # goal None: the free-text path (ask)
    if dispatch.kind != "plan":
        return dict(dispatch=dispatch.kind, reason=dispatch.reason)
    decision, read = agent.plan(agent.view, dispatch.goal, presses_done=0)
    return dict(dispatch="plan", option=decision.option, action=list(decision.action or ()),
                value=round(decision.value, 6), read=[list(c) for c in read.components])


def visual_training_context(identity_run):
    """Disclose parent diagnostics separately from the actual runtime task gates."""
    parent = Path(identity_run)
    record = json.loads((parent / 'run.json').read_text())
    frozen = record['identity']['settings'].get('freeze_perception', False)
    result = json.loads((parent / 'result.json').read_text()) if frozen else None
    return dict(parent=str(parent), frozen_perception_key_repair=frozen,
                parent_result_sha256=file_hash(parent / 'result.json') if frozen else None,
                training_monitor_gate=result.get('gate') if result else None,
                protocol='The added cosine-quantile training screen is a reported diagnostic, not a runtime-task prerequisite. '
                         'This prospective amendment was recorded before calibration3404 and validation2405/2406 because '
                         'the proxy cutoffs were not derived from the binder. Failed parent screens remain failed; '
                         'original actual-task gates are unchanged. See docs/real-visual-memory-plan.md.')


LEGACY_MATCH_THRESHOLDS = (.80, .85, .90, .95)
LEGACY_RULE = ('Actual session grid: match in [.8,.85,.9,.95], new=match-.05, margin in [.05,.1]; '
               'maximize min acquisition/same-layout/relocated matching under false-match<=.005')


def checked_match_thresholds(values):
    """Calibration match grid: non-empty, finite, strictly increasing numbers, each valid
    for AssociationBinder with new = match - .05 (-1 <= new < match <= 1)."""
    values = tuple(values)
    if (not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
            or any(not math.isfinite(v) or not -1 <= v - .05 or v > 1 for v in values)
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError('Binding match thresholds must be finite, strictly increasing, in [-0.95, 1]')
    return tuple(float(v) for v in values)


@torch.no_grad()
def calibrate_visual_binding(output, *, identity_run, seed=3401, scenes=128, match_thresholds=LEGACY_MATCH_THRESHOLDS):
    """Calibrate the existing machine-key policy on training scenes only; no updates.

    Hidden machine pixels supply calibration labels only. Runtime eligibility uses
    the learned kind head. Every policy is measured in the actual sequential session.
    `match_thresholds` is the searched match grid (new = match - .05, margins .05/.1);
    the default is the legacy grid and leaves settings unchanged.
    """
    from pathwm.models.slots import pointer
    if identity_run is None or scenes < 1:
        raise ValueError('Calibration needs an identity checkpoint and positive scene count')
    grid = checked_match_thresholds(match_thresholds)
    legacy = grid == LEGACY_MATCH_THRESHOLDS
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    modules = build(seed, identity_run=identity_run, candidate_scope="predicted-machine")
    before = {k: state_hash(v) for k, v in modules.items()}
    perception, candidates = modules['perception'], modules['candidates']
    settings = dict(stage='visual-binding-calibration', purpose='development', seed=seed,
                    scenes=scenes, device='cpu', precision='fp32', width=WIDTH, updates=0,
                    training_context=visual_training_context(identity_run),
                    episode='initial, same-layout lamp change, relocated with alternate side swaps, novel arrival',
                    checkpoint_sha256=file_hash(Path(identity_run)/'last.pt'),
                    rule=LEGACY_RULE if legacy else LEGACY_RULE.replace('[.8,.85,.9,.95]', str(list(grid))))
    if not legacy:
        settings['match_thresholds'] = list(grid)
    atomic_json(out/'status.json', dict(result='running',report='pending',step=0,error=None))
    atomic_json(out/'run.json',dict(schema='pathwm-run-v1',identity=dict(settings=settings,
                data=dict(population='train',kinds=list(rw.KIND_SPLIT['train'])),environment=environment('cpu')),
                source=source_record(__file__,nn.ModuleDict(modules))))
    (out/'recipe.py').write_text(Path(__file__).read_text())
    g = torch.Generator().manual_seed(seed)
    kinds = torch.tensor(rw.KIND_SPLIT['train'])
    rows, positives, negatives, episodes = [], [], [], []
    for index in range(scenes):
        scene = rw.sample_scenes(g,kinds[torch.randperm(len(kinds),generator=g)[:2]][None])
        lamps = torch.randint(2,(1,2),generator=g)
        changed = lamps.clone(); changed[0,0] = 1-changed[0,0]
        swapped = index % 2 == 1
        other_scene = rw.sample_scenes(g,scene.kind.flip(1) if swapped else scene.kind)
        moved_lamps = changed.flip(1) if swapped else changed
        remaining = kinds[~torch.isin(kinds,scene.kind.flatten())]
        novel_kind = remaining[torch.randint(len(remaining),(1,),generator=g)].item()
        novel_scene = rw.sample_scenes(g,[[novel_kind,int(other_scene.kind[0,1])]])
        episodes.append((scene,lamps,other_scene,changed,moved_lamps,novel_scene))
        frames = torch.cat((rw.render(scene,lamps)[0],rw.render(other_scene,moved_lamps)[0]))
        percept = perception(frames)
        keys, _ = candidates(percept.slots)
        # Same true machine in two views (calibration supervision only).
        indices = torch.stack([pointer(percept.alpha,torch.cat((scene.machine_xy[:,side],other_scene.machine_xy[:,side])))
                               for side in range(2)],1)
        selected = keys[torch.arange(2)[:,None],indices]
        if swapped:
            selected = torch.stack((selected[0], selected[1].flip(0)))
        same = (selected[0]*selected[1]).sum(-1)
        different = selected[:,0] @ selected[:,1].T
        positives.extend(same.tolist()); negatives.extend(different.flatten().tolist())
        predicted = percept.kind.argmax(-1)
        detection = bool((predicted.gather(1,indices)==1).all() and (indices[:,0]!=indices[:,1]).all())
        row=dict(step=index,split='calibration',same_min=float(same.min()),
                 different_max=float(different.max()),detected=detection,
                 kinds=scene.kind.tolist(),swapped=swapped,same_scores=same.tolist(),different_scores=different.tolist())
        rows.append(row)
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n')
    policies = []
    for match in grid:
        for margin in (.05,.10):
            thresholds = dict(match_threshold=match,new_threshold=match-.05,margin=margin)
            binder = AssociationBinder(**thresholds)
            acquired = matched = same_matched = same_false = false = novel_false = novel_slot_detected = 0
            policy_index = len(policies)
            for index,(scene,lamps,other_scene,changed,moved_lamps,novel_scene) in enumerate(episodes):
                agent = new_agent(modules,out/f'policy-{policy_index}'/f'case-{index:03d}',binder=binder)
                first_rgb,first_labels = rw.render(scene,lamps)
                first,_ = agent.observe(first_rgb[0])
                owners = {}
                for m in first.machines:
                    x,y = (int(v) for v in m['xy'])
                    label = int(first_labels[0,y,x])-1
                    if m['instance'] is not None and label in (0,1):
                        owners[m['instance']] = int(scene.kind[0,label])
                acquired += len(set(owners.values()))
                # Match evaluation's observation history before relocation.
                changed_rgb,changed_labels = rw.render(scene,changed)
                same_view,_ = agent.observe(changed_rgb[0])
                for m in same_view.machines:
                    x,y = (int(v) for v in m['xy'])
                    label = int(changed_labels[0,y,x])-1
                    if m['instance'] in owners and label in (0,1):
                        correct = owners[m['instance']] == int(scene.kind[0,label])
                        same_matched += int(correct); same_false += int(not correct)
                next_rgb,next_labels = rw.render(other_scene,moved_lamps)
                second,_ = agent.observe(next_rgb[0])
                for m in second.machines:
                    x,y = (int(v) for v in m['xy'])
                    label = int(next_labels[0,y,x])-1
                    if m['instance'] in owners and label in (0,1):
                        correct = owners[m['instance']] == int(other_scene.kind[0,label])
                        matched += int(correct); false += int(not correct)
                known = {e.id for e in agent.session.store.entities() if e.kind == 'instance'}
                novel_rgb,novel_labels = rw.render(novel_scene,moved_lamps)
                arrival,_ = agent.observe(novel_rgb[0])
                novel_hits = [m for m in arrival.machines
                              if int(novel_labels[0,int(m['xy'][1]),int(m['xy'][0])]) == 1]
                novel_slot_detected += int(bool(novel_hits))
                novel_false += int(any(m['instance'] in known for m in novel_hits))
            result = dict(thresholds=thresholds,acquisition=acquired/(2*scenes),
                          correct_matching=matched/(2*scenes),same_layout_matching=same_matched/(2*scenes),
                          false_matches=max(false,same_false)/(2*scenes),
                          same_layout_false_matches=same_false/(2*scenes),
                          relocated_false_matches=false/(2*scenes),
                          novel_false_merge=novel_false/scenes,novel_slot_detection=novel_slot_detected/scenes)
            policies.append(result)
            print(json.dumps(result),flush=True)
    admissible = [p for p in policies if p['false_matches'] <= .005 and p['novel_false_merge'] <= .005 and p['novel_slot_detection'] >= .95]
    selected = max(admissible,key=lambda p:(min(p['acquisition'],p['same_layout_matching'],p['correct_matching']),
                    -max(p['false_matches'],p['novel_false_merge']),p['thresholds']['match_threshold'],p['thresholds']['margin']),default=None)
    unchanged = before == {k:state_hash(v) for k,v in modules.items()}
    gate = bool(selected and min(selected['acquisition'],selected['same_layout_matching'],selected['correct_matching']) >= .95 and unchanged)
    metrics=dict(min_same=min(positives),max_different=max(negatives),
                 detected=sum(r['detected'] for r in rows)/scenes,weights_unchanged=unchanged,
                 selected=selected,policies=policies)
    atomic_json(out/'policies.json',policies)
    atomic_json(out/'result.json',dict(evaluation_scope='Training-only calibration of existing machine identity policy',
                gate=gate,metrics=metrics,training_context=settings['training_context'],
                limitations=['No network updates; bounded two-machine layout/lamp-change task.',
                'Hidden identity used only for calibration labels; runtime uses learned kind eligibility.',
                'Thresholds are unqualified until the independent validation screen passes.']))
    atomic_json(out/'status.json',dict(result='completed',report='pending',step=0,error=None))
    if gate:
        atomic_json(out/'binding.json',dict(schema='pathwm-visual-binding-v2',
                    checkpoint_sha256=settings['checkpoint_sha256'],candidate_scope='predicted-machine',
                    thresholds=selected['thresholds'],
                    perception_sha256=state_hash(perception),key_sha256=state_hash(candidates.key),
                    source_sha256=source_record(__file__,nn.ModuleDict(modules))['sha256'],
                    result_sha256=file_hash(out/'result.json'),policies_sha256=file_hash(out/'policies.json'),
                    calibration=dict(seed=seed,scenes=scenes,population='train',method='actual-session-grid-with-novel-arrivals',
                                     **({} if legacy else dict(match_thresholds=list(grid))))))
    write_report(out)
    print(json.dumps(dict(gate=gate,metrics=metrics),indent=2))
    return gate


def visual_binding_policy(path, identity_run):
    if path is None:
        return None, None
    if identity_run is None:
        raise ValueError('Binding calibration requires --identity-run')
    policy = json.loads(Path(path).read_text())
    if (policy.get('schema')!='pathwm-visual-binding-v2'
            or policy.get('checkpoint_sha256')!=file_hash(Path(identity_run)/'last.pt')
            or policy.get('candidate_scope')!='predicted-machine'):
        raise ValueError('Binding calibration differs from the loaded visual model or candidate scope')
    directory = Path(path).parent
    result = json.loads((directory/'result.json').read_text())
    status = json.loads((directory/'status.json').read_text())
    policies = json.loads((directory/'policies.json').read_text())
    model = build(0,identity_run=identity_run,candidate_scope=policy['candidate_scope'])
    if (not result.get('gate') or status.get('result') != 'completed'
            or file_hash(directory/'result.json') != policy.get('result_sha256')
            or file_hash(directory/'policies.json') != policy.get('policies_sha256')
            or source_record(__file__,nn.ModuleDict(model))['sha256'] != policy.get('source_sha256')
            or state_hash(model['perception']) != policy.get('perception_sha256')
            or state_hash(model['candidates'].key) != policy.get('key_sha256')
            or result['metrics']['selected']['thresholds'] != policy['thresholds']
            or result['metrics']['selected'] not in policies):
        raise ValueError('Binding calibration evidence, model or source identity changed')
    grid = policy['calibration'].get('match_thresholds')
    if grid is not None and (sorted({p['thresholds']['match_threshold'] for p in policies}) != grid
                             or policy['thresholds']['match_threshold'] not in grid):
        raise ValueError('Binding calibration grid changed')
    return AssociationBinder(**policy['thresholds']), policy


@torch.no_grad()
def visual_memory_evaluation(output, *, identity_run, seed, scenes, binding_calibration=None):
    """Native J visual-memory connection; labels are evaluation-only.

    No optimizer, supplied candidate keys, new representation or surrogate decoder.
    This is a development screen, not fine-detail or full-agent qualification.
    """
    import time
    from pathwm.world_state.concepts import SOURCE

    if identity_run is None or scenes < 1:
        raise ValueError('Visual-memory evaluation needs --identity-run and positive --scenes')
    started = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'status.json', dict(result='running', report='pending', step=0, error=None))
    binder, policy = visual_binding_policy(binding_calibration, identity_run)
    modules = build(seed, identity_run=identity_run, candidate_scope=policy["candidate_scope"] if policy else "all")
    perception = modules['perception']
    initial = {k: state_hash(v) for k, v in modules.items()}
    source = source_record(__file__, nn.ModuleDict(modules))
    settings = dict(stage='actual-visual-memory', purpose='development', device='cpu',
                    precision='fp32', seed=seed, scenes=scenes, width=WIDTH,
                    slots=7, iterations=3, decoder_width=32,
                    identity_run=str(identity_run), training_context=visual_training_context(identity_run),
                    checkpoint_sha256=file_hash(Path(identity_run) / 'last.pt'),
                    screen_threshold=.95, updates=0, binding_policy=policy,
                    candidate_scope=policy['candidate_scope'] if policy else 'all',
                    scope='Actual native perception/slot/decoder and R2 memory; other R2 modules untrained',
                    example_labels={'input': 'Machine pixels (evaluation-only mask)', 'rgb': 'Actual memory render (same evaluation mask; lossy)'})
    atomic_json(out / 'run.json', dict(schema='pathwm-run-v1', identity=dict(settings=settings,
                data=dict(population='validation', kinds=list(rw.KIND_SPLIT['validation']),
                          generator='rule_world_64_v1; independent seeded fresh scenes'),
                environment=environment('cpu')), source=source, initial_model_sha256=initial))
    (out / 'recipe.py').write_text(Path(__file__).read_text())
    save_models(modules, out / 'models.pt')
    g = torch.Generator().manual_seed(seed)
    kinds = torch.tensor(rw.KIND_SPLIT['validation'])
    rows, examples, reconstructions = [], [], []
    try:
        for index in range(scenes):
            scene = rw.sample_scenes(g, kinds[torch.randperm(len(kinds), generator=g)[:2]][None])
            lamps = torch.randint(2, (1, 2), generator=g)
            rgb, labels = rw.render(scene, lamps)
            directory = out / f'case-{index:03d}'
            agent = new_agent(modules, directory, binder=binder)
            view, result = agent.observe(rgb[0])  # Actual learned candidates; no label input.
            row = dict(step=index, split='validation', scene=index,
                       acquired=0, matched=0, correct_lamps=0, checks_pass=True)
            old, old_by_side = {}, {}
            colors, alpha = perception.decoder(view.percept.slots)
            for side, machine in enumerate(view.machines):
                instance, slot = machine['instance'], machine['slot']
                if instance is None:
                    continue
                try:
                    component = agent.visual_memory(instance)
                except LookupError:
                    continue
                if component is None:
                    continue
                decoded = agent.render_memory(instance)
                exact = torch.equal(component.tensor(), view.percept.slots[0, slot].cpu())
                live_rgb, live_alpha = perception.decoder(view.percept.slots[:, slot:slot+1])
                exact = exact and torch.equal(decoded['rgb'], live_rgb) and torch.equal(decoded['alpha'], live_alpha)
                row[f'batch_rgb_difference_{side}'] = float((decoded['rgb'][0,0]-colors[0,slot]).abs().max())
                row[f'batch_alpha_difference_{side}'] = float((decoded['alpha'][0,0]-alpha[0,slot]).abs().max())
                retained_source = agent.source_pyramid(instance)
                live_source = perception.pyramid(rgb)
                exact = exact and len(retained_source.scales) == len(live_source.scales)
                for a, b in zip(retained_source.scales, live_source.scales):
                    exact = exact and all((getattr(a, n) is None and getattr(b, n) is None) or
                                          (isinstance(getattr(a, n), torch.Tensor) and isinstance(getattr(b, n), torch.Tensor)
                                           and torch.equal(getattr(a, n), getattr(b, n)))
                                          for n in ('values', 'times', 'valid', 'ends', 'content_times'))
                    exact = exact and a.grid == b.grid
                row['checks_pass'] &= exact
                x, y = (int(v) for v in machine['xy'])
                row['acquired'] += int(labels[0, y, x] == side + 1)
                old[instance] = component
                old_by_side[side] = instance
                mask = labels[0] == side + 1
                row[f'machine_{side}_mse'] = float((decoded['rgb'][0, 0] - rgb[0]).square()[:, mask].mean())
                row['correct_lamps'] += int(int(labels[0,y,x]) == side+1 and (decoded['lamp'] > 0).long().item() == lamps[0, side].item())
                if index < 4:
                    examples.append(rgb[0]*mask[None])
                    reconstructions.append(decoded['rgb'][0,0]*mask[None])
                other = view.machines[1-side]['slot'] if len(view.machines) == 2 else slot
                row[f'wrong_memory_{side}_mse'] = float((colors[0, other] - rgb[0]).square()[:, mask].mean())
            atomic_json(directory / 'initial-decisions.json', result)
            row['reconstruction_mse'] = float((view.percept.recon - rgb).square().mean())
            # Change one source lamp; hidden labels still go only to rendering/evaluation.
            changed = lamps.clone(); changed[0, 0] = 1 - changed[0, 0]
            updated, _ = rw.render(scene, changed)
            newer, correction = agent.observe(updated[0])
            atomic_json(directory / 'updated-decisions.json', correction)
            current = {}
            row['updated_lamps'] = 0
            for side, machine in enumerate(newer.machines):
                instance = machine['instance']
                if instance is None:
                    continue
                try:
                    c = agent.visual_memory(instance)
                except LookupError:
                    continue
                if c is None:
                    continue
                decoded = agent.render_memory(instance)
                current[instance] = (c, decoded['rgb'].clone())
                row['matched'] += int(instance == old_by_side.get(side))
                row['updated_lamps'] += int(instance == old_by_side.get(side) and (decoded['lamp'] > 0).long().item() == changed[0, side].item())
                row['checks_pass'] &= c.data['event'] == newer.event_id
                row['checks_pass'] &= torch.equal(c.tensor(), newer.percept.slots[0, machine['slot']].cpu())
            row['checks_pass'] &= all(agent.session.store.component(c.id) == c for c in old.values())
            swapped = index % 2 == 1
            next_scene = rw.sample_scenes(g, scene.kind.flip(1) if swapped else scene.kind)
            next_lamps = changed.flip(1) if swapped else changed
            moved, moved_labels = rw.render(next_scene, next_lamps)
            final_view, final_observation = agent.observe(moved[0])
            atomic_json(directory / 'relocated-decisions.json', final_observation)
            row['cross_layout_matches'] = 0
            row['cross_layout_lamps'] = 0
            current = {}
            for side, machine in enumerate(final_view.machines):
                instance = machine['instance']
                if instance is None:
                    continue
                original_side = 1-side if swapped else side
                x,y = (int(v) for v in machine['xy'])
                correct = instance == old_by_side.get(original_side) and int(moved_labels[0,y,x]) == side+1
                row['cross_layout_matches'] += int(correct)
                try:
                    c = agent.visual_memory(instance)
                except LookupError:
                    continue
                decoded = agent.render_memory(instance)
                row['cross_layout_lamps'] += int(correct and (decoded['lamp']>0).long().item()==next_lamps[0,side].item())
                row['checks_pass'] &= c.data['event'] == final_view.event_id
                row['checks_pass'] &= torch.equal(c.tensor(),final_view.percept.slots[0,machine['slot']])
                current[instance] = (c,decoded['rgb'].clone())
            # Independent source-owned record must survive another observation's withdrawal.
            agent.save(directory)
            snapshot_before_novel = agent.session.store.snapshot()
            known = {e.id for e in agent.session.store.entities() if e.kind == 'instance'}
            remaining = kinds[~torch.isin(kinds,next_scene.kind.flatten())]
            novel_kind = remaining[torch.randint(len(remaining),(1,),generator=g)].item()
            novel_scene = rw.sample_scenes(g,[[novel_kind,int(next_scene.kind[0,1])]])
            novel_rgb,novel_labels = rw.render(novel_scene,next_lamps)
            arrival,arrival_result = agent.observe(novel_rgb[0])
            atomic_json(directory/'novel-decisions.json',arrival_result)
            row['novel_slot_detected'] = False
            row['novel_false_merge'] = False
            row['novel_new'] = False
            for m in arrival.machines:
                x,y = (int(v) for v in m['xy'])
                if int(novel_labels[0,y,x]) == 1:
                    row['novel_slot_detected'] = True
                    row['novel_false_merge'] |= m['instance'] in known
                    row['novel_new'] |= m['instance'] is not None and m['instance'] not in known
            restored = restore_agent(load_models(out / 'models.pt', seed=seed+1), directory, binder=binder)
            row['restart_equal'] = snapshot_before_novel == restored.session.store.snapshot()
            expected_pyramid = perception.pyramid(moved)
            restored.resume_view()
            for instance, (component, rendered) in current.items():
                row['restart_equal'] &= restored.visual_memory(instance) == component
                row['restart_equal'] &= torch.equal(restored.render_memory(instance)['rgb'], rendered)
                actual_pyramid = restored.source_pyramid(instance)
                row['restart_equal'] &= all(torch.equal(a.values, b.values)
                    for a, b in zip(actual_pyramid.scales, expected_pyramid.scales))
            frame = final_observation['evidence'][0]
            tx = restored.session.store.begin(f'withdraw-{index}', occurred_at=restored.session.time,
                    available_at=restored.session.time + 1, kind='correction', payload=dict(source=SOURCE))
            tx.retract_evidence(frame)
            restored.correct(tx)
            rejected = 0
            for instance in current:
                try:
                    recalled = restored.visual_memory(instance)
                    rejected += int(recalled is None)
                except LookupError:
                    rejected += 1
            row['retracted_reads_rejected'] = rejected == len(current) and len(current) > 0
            row['frame_identity_withdrawn'] = all(not restored.session.store.component(c.parents[0]).active
                                                  for c,_ in current.values())
            row['checks_pass'] &= row['frame_identity_withdrawn']
            # Earlier records remain byte-for-byte immutable, but never auto-selected as current.
            row['old_records_unchanged'] = all(restored.session.store.component(c.id) == c for c in old.values())
            row['checks_pass'] &= row['restart_equal'] and row['retracted_reads_rejected'] and row['old_records_unchanged']
            rows.append(row)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, sort_keys=True) + '\n')
            print(f'visual memory {index+1}/{scenes}: acquired={row["acquired"]}, matched={row["matched"]}, checks={row["checks_pass"]}', flush=True)
        unchanged = initial == {k: state_hash(v) for k, v in modules.items()}
        count = scenes * 2
        metrics = dict(acquisition=sum(r['acquired'] for r in rows)/count,
                       identity_matching=sum(r['matched'] for r in rows)/count,
                       lamp_accuracy=sum(r['correct_lamps'] for r in rows)/count,
                       updated_lamp_accuracy=sum(r['updated_lamps'] for r in rows)/count,
                       cross_layout_matching=sum(r['cross_layout_matches'] for r in rows)/count,
                       cross_layout_lamp_accuracy=sum(r['cross_layout_lamps'] for r in rows)/count,
                       novel_slot_detection=sum(r['novel_slot_detected'] for r in rows)/scenes,
                       novel_false_merge=sum(r['novel_false_merge'] for r in rows)/scenes,
                       novel_acquisition=sum(r['novel_new'] for r in rows)/scenes,
                       reconstruction_mse=sum(r['reconstruction_mse'] for r in rows)/scenes,
                       all_contracts=all(r['checks_pass'] for r in rows), weights_unchanged=unchanged,
                       seconds=time.perf_counter()-started)
        gate = metrics['all_contracts'] and unchanged and metrics['novel_false_merge'] <= .005 and min(metrics[k] for k in
                    ('acquisition','identity_matching','lamp_accuracy','updated_lamp_accuracy','cross_layout_matching','cross_layout_lamp_accuracy','novel_slot_detection')) >= .95
        atomic_json(out / 'result.json', dict(evaluation_scope=settings['scope'], gate=gate, metrics=metrics,
                    training_context=settings['training_context'],
                    limitations=['Synthetic validation scenes; development screen, not sealed generalization.',
                      'Slots are compressed and position-dependent; no fine-detail or pose-transfer claim.',
                      'Full pyramid access re-encodes retained lossless source; slot recall does not.',
                      'Other R2 core/belief/action/value modules remain untrained; J is provisional.',
                      'Novel-slot detection is gated; novel identity acquisition is reported, not gated. Abstention is allowed.',
                      'After frame withdrawal, affected instances are not re-identified; later sightings create new instances (no replay/reconciliation).',
                      'Wrong-memory RGB comparison includes position differences and is diagnostic only.',
                      'The old surrogate MSE0.002 target is not qualified by output parity.']))
        atomic_json(out / 'status.json', dict(result='completed', report='pending', step=0, error=None))
    except Exception as error:
        atomic_json(out / 'status.json', dict(result='failed', report='pending', step=0, error=str(error)))
        raise
    try:
        write_report(out, batch={'rgb':torch.stack(examples)}, outputs={'rgb':torch.stack(reconstructions)})
    except Exception as error:
        atomic_json(out / 'status.json', dict(result='completed', report='failed', step=0, error=str(error)))
        raise
    print(json.dumps(dict(gate=gate, metrics=metrics), indent=2))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--perception", type=Path, help="R1 perception run to initialise slots (read-only)")
    parser.add_argument("--learned-keys", action="store_true",
                        help="bind with the (untrained) learned slot keys instead of the pixel fixture")
    parser.add_argument("--identity-run", type=Path,
                        help="S1 run trained with --identity: load its perception and exported key together; "
                             "use --binding-calibration for calibrated thresholds; no full-agent claim")
    parser.add_argument("--visual-memory", action="store_true", help="evaluate native J visual memory and restart")
    parser.add_argument("--calibrate-visual-binding", action="store_true", help="calibrate existing J machine identity policy on train scenes")
    parser.add_argument("--binding-calibration", type=Path, help="checkpoint-bound binding.json for visual memory")
    parser.add_argument("--binding-match-thresholds", type=float, nargs="+",
                        help="with --calibrate-visual-binding: searched match grid (strictly increasing; "
                             "default legacy .80 .85 .90 .95; new = match - .05; margins .05/.10)")
    args = parser.parse_args()
    if args.binding_match_thresholds is not None:
        if not args.calibrate_visual_binding:
            parser.error("--binding-match-thresholds requires --calibrate-visual-binding")
        try:
            checked_match_thresholds(args.binding_match_thresholds)
        except ValueError as error:
            parser.error(str(error))
    torch.set_num_threads(2)
    if args.calibrate_visual_binding:
        grid = args.binding_match_thresholds or LEGACY_MATCH_THRESHOLDS
        passed = calibrate_visual_binding(args.output, identity_run=args.identity_run, seed=args.seed, scenes=args.scenes,
                                          match_thresholds=grid)
        if not passed:
            raise SystemExit(1)
        return
    if args.visual_memory:
        passed = visual_memory_evaluation(args.output, identity_run=args.identity_run, seed=args.seed, scenes=args.scenes, binding_calibration=args.binding_calibration)
        if not passed:
            raise SystemExit(1)
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.jsonl").write_text("")
    atomic_json(out / "status.json", dict(result="running", report="pending", step=0, error=None))
    perception = None if args.perception is None else dict(
        run=str(args.perception), checkpoint_sha256=file_hash(args.perception / "last.pt"))
    if args.identity_run is not None:
        perception = dict(identity_run=str(args.identity_run),
                          checkpoint_sha256=file_hash(args.identity_run / "last.pt"),
                          key="exported S1 identity key = core.key_head = session candidate key; "
                              "full provenance in models.json")
    _, binding_policy = visual_binding_policy(args.binding_calibration, args.identity_run)
    run = describe(args.perception, args.identity_run, args.learned_keys, calibrated=binding_policy is not None)
    settings = dict(stage="unified-session-life", purpose="software", device="cpu", seed=args.seed,
                    scenes=args.scenes, width=WIDTH, perception=perception, binding=run["binding"],
                    binding_policy=binding_policy, candidate_scope=binding_policy["candidate_scope"] if binding_policy else "all",
                    weights="random (seeded) except a supplied perception/key checkpoint", scope=run["scope"])
    atomic_json(out / "run.json", dict(schema="pathwm-run-v1", identity=dict(
        settings=settings, data=dict(kinds=list(rw.KIND_SPLIT["train"][:2]), rules="train split, seeded"),
        environment=environment("cpu")), source=source_record(__file__, nn.Module())))
    (out / "recipe.py").write_text(Path(__file__).read_text())
    try:
        rows, restart, final = life(out, seed=args.seed, scenes=args.scenes, perception_run=args.perception,
                                    fixture=not (args.learned_keys or args.identity_run),
                                    identity_run=args.identity_run, binding_calibration=args.binding_calibration)
        with (out / "metrics.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps({k: v for k, v in row.items() if k != "steps"}, sort_keys=True) + "\n")
        atomic_json(out / "life.json", dict(rows=rows, restart=restart, final=final))
        goals = [r for r in rows if r["kind"] == "goal"]
        atomic_json(out / "result.json", dict(
            evaluation_scope=run["scope"], gate=None,
            metrics=dict(final=final, restart_equal=bool(restart and restart["equal"]),
                         free_text_asked=all(r["option"] == "ask" and r["actuator_presses"] == 0
                                             for r in rows if r["kind"] == "free_text"),
                         outcomes={r["outcome"]: sum(g["outcome"] == r["outcome"] for g in goals) for r in goals},
                         # Measured, not claimed: what this life actually exercised.
                         exercised=dict(identity_matches=final["decisions"]["matched"],
                                        concepts=final["concepts"], attributed=final["attributed"],
                                        agent_presses=final["agent_presses"],
                                        agent_ok_presses=final["agent_ok_presses"],
                                        success_after_press=sum(g["outcome"] == "success" and g["presses"] > 0
                                                                for g in goals),
                                        goals_planned=sum(g["option"] not in ("ask", "no_identified_machine",
                                                                              "unsupported") for g in goals))),
            limitations=run["limitations"],
        ))
    except Exception as error:
        atomic_json(out / "status.json", dict(result="failed", report="incomplete", step=0,
                                              error=f"{type(error).__name__}: {error}"))
        raise
    atomic_json(out / "status.json", dict(result="completed", report="pending", step=0, error=None))
    try:
        write_report(out)
    except Exception as error:
        atomic_json(out / "status.json", dict(result="completed", report="failed", step=0,
                                              error=f"report: {type(error).__name__}: {error}"))
        raise
    print(json.dumps(dict(final=final, restart_equal=restart and restart["equal"]), indent=1))


if __name__ == "__main__":
    main()
