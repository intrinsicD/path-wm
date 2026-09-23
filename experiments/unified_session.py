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


def describe(perception_run=None, identity_run=None, learned_keys=False):
    """Scope, binding and limitations of THIS run's actual composition (no overclaim)."""
    if identity_run is not None:
        loaded = "perception and exported identity key from one S1 --identity run (named in run.json/models.json)"
        binding = "exported S1 identity key (uncalibrated binder thresholds)"
        keys = ("Session identity uses the exported S1 key (= core.key_head) on slot tokens with untrained, "
                "uncalibrated binder thresholds: identity decisions are a software exercise, not learned tracking.")
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


def build(seed, perception_run=None, identity_run=None, *, shared_key=False):
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
    modules = build(seed, shared_key=source["source"] != "own_projection")
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
         identity_run=None):
    """Persistent machines across `scenes` layouts; restart after `restart_after`.

    fixture=True binds with `pixel_candidates` (SOFTWARE); False uses the learned
    (here untrained) slot candidate encoder."""
    directory = Path(directory)
    g = torch.Generator().manual_seed(seed)
    modules = build(seed, perception_run, identity_run)
    save_models(modules, directory / "models.pt")
    memory_dir = directory / "memory"
    agent = new_agent(modules, memory_dir)
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
            agent = restore_agent(modules, memory_dir)
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
                             "the session binds with that key (uncalibrated thresholds; no tracking claim)")
    args = parser.parse_args()
    torch.set_num_threads(2)
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
    run = describe(args.perception, args.identity_run, args.learned_keys)
    settings = dict(stage="unified-session-life", purpose="software", device="cpu", seed=args.seed,
                    scenes=args.scenes, width=WIDTH, perception=perception, binding=run["binding"],
                    weights="random (seeded) except a supplied perception/key checkpoint", scope=run["scope"])
    atomic_json(out / "run.json", dict(schema="pathwm-run-v1", identity=dict(
        settings=settings, data=dict(kinds=list(rw.KIND_SPLIT["train"][:2]), rules="train split, seeded"),
        environment=environment("cpu")), source=source_record(__file__, nn.Module())))
    (out / "recipe.py").write_text(Path(__file__).read_text())
    try:
        rows, restart, final = life(out, seed=args.seed, scenes=args.scenes, perception_run=args.perception,
                                    fixture=not (args.learned_keys or args.identity_run),
                                    identity_run=args.identity_run)
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
