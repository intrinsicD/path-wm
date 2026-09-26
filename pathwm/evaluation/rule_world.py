"""R1 evaluation: frozen-weight pixel lives, controls, metrics and gates.

The harness owns scenes, hidden rules and truth labels. The agent receives only
frames, action records/receipts, goals and source correction messages. Truth is
used after the fact for scoring. Controls and diagnostics are labelled as such.
"""

from dataclasses import dataclass, replace
import math
import time

import numpy as np
import torch

from pathwm.data import rule_world as rw
from pathwm.io import state_hash
from pathwm.models import latent_core as lc
from pathwm.models.slots import ATTRIBUTES, VALUES, pointer
from pathwm.world_state.concepts import AgentSettings, ConceptAgent, ConceptMemory  # noqa: F401 (AgentSettings re-exported)

# Generator-side family schedule for the queried rules (A, B) of the declared four
# lives per support level. Fixed before any model runs; never chosen by performance.
# Every family/operator appears at each support level (category, relation and toggle
# twice; open and close once) and the schedule is identical across seeds/controls.
FAMILY_SCHEDULE = (
    ("category", "relation"),
    ("open", "close"),
    ("toggle", "category"),
    ("relation", "toggle"),
)

T_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}
TRANSITION_FAMILIES = ("open", "close", "toggle")


# ---------------------------------------------------------------- training tokens


def encode_episodes(perceive, batch, device, chunk=256):
    """Render each scene once per lamp configuration, perceive, assemble tokens.

    `perceive(scenes, lamps)` -> Percept. Demonstration coordinates are exact
    scene centers (the logged action record); slot choice uses the percept's own
    alpha. Post targets are the perceived post-frame target-machine slots.
    """
    scenes = batch.scenes
    count = len(scenes)
    configs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    index = torch.arange(count).repeat_interleave(4)
    lamps = configs.repeat(count, 1)
    slots, alphas, lamp_logits = [], [], []
    with torch.no_grad():
        for start in range(0, len(index), chunk):
            part = index[start : start + chunk]
            p = perceive(_scenes_to(scenes.select(part), device), lamps[start : start + chunk].to(device))
            slots.append(p.slots)
            alphas.append(p.alpha)
            lamp_logits.append(p.lamp)
    slots = torch.cat(slots)
    alpha = torch.cat(alphas)
    machine_xy = scenes.machine_xy.to(device)
    object_xy = scenes.object_xy.to(device)

    def frame(scene, lamps):
        return scene * 4 + lamps[:, 0] * 2 + lamps[:, 1]

    def gather(t, object_lamps=None):
        scene = t.scene.to(device)
        m = t.machine.to(device)
        pre = frame(scene, t.pre.to(device))
        post = frame(scene, t.post.to(device))
        # Rollouts read objects from the plan's initial observation, never from
        # intermediate observations that a planner would not have.
        objects = pre if object_lamps is None else frame(scene, object_lamps.to(device))
        m_xy = machine_xy[scene, m]
        ptr_m = pointer(alpha[pre], m_xy)
        ptr_a = pointer(alpha[objects], object_xy[scene, t.a.to(device)])
        ptr_b = pointer(alpha[objects], object_xy[scene, t.b.to(device)])
        ptr_post = pointer(alpha[post], m_xy)
        return lc.TransitionTokens(
            slots[pre, ptr_m], slots[objects, ptr_a], slots[objects, ptr_b],
            slots[post, ptr_post], t.outcome.float().to(device),
            t.episode.to(device), t.step.to(device),
        )

    episodes = len(batch.target_kind)
    keys = []
    for e in range(episodes):
        own = (batch.episode_of_scene == e).nonzero().flatten()[:2]
        f = frame(own.to(device), torch.zeros(len(own), 2, dtype=torch.long, device=device))
        xy = machine_xy[own.to(device), batch.target_machine[own].to(device)]
        keys.append(slots[f, pointer(alpha[f], xy)])
    chain = batch.chain
    initial = {int(e): chain.pre[k] for k, e in enumerate(chain.episode) if int(chain.step[k]) == 0}
    chain_objects = torch.stack([initial[int(e)] for e in chain.episode]) if len(chain) else None
    return lc.EpisodeTokens(
        gather(batch.support), gather(batch.query), gather(chain, chain_objects),
        torch.stack(keys), episodes, batch.target_kind.to(device),
    )


def _scenes_to(scenes, device):
    return rw.Scenes(*(v.to(device) for v in vars(scenes).values()))


def pixel_perceiver(perception):
    def perceive(scenes, lamps):
        rgb, _ = rw.render(scenes, lamps)
        return perception(rgb)

    return perceive


def symbolic_perceiver(symbolic):
    def perceive(scenes, lamps):
        _, entity = rw.render(scenes, lamps)
        return symbolic(scenes, lamps, entity)

    return perceive


@torch.no_grad()
def symbolic_episode_tokens(symbolic, batch, device):
    """`encode_episodes(symbolic_perceiver(symbolic), ...)` without its discarded rendering.

    Diagnostic supplied-symbol path only (scene attributes and true lamp states, as
    `SymbolicSlots` already receives): machine tokens are lamp+role embeddings and
    object tokens do not depend on lamps. Bit-identical to the rendered path (tested).
    """
    attrs = batch.scenes.attrs.to(device)
    objects = symbolic.attribute(attrs + torch.arange(ATTRIBUTES, device=device) * VALUES).sum(2)
    objects = objects + symbolic.role.weight[2]
    machines = symbolic.lamp_embedding.weight + symbolic.role.weight[1]

    def gather(t):
        scene, m = t.scene.to(device), t.machine.to(device)
        row = torch.arange(len(t), device=device)
        return lc.TransitionTokens(
            machines[t.pre.to(device)[row, m]], objects[scene, t.a.to(device)], objects[scene, t.b.to(device)],
            machines[t.post.to(device)[row, m]], t.outcome.float().to(device),
            t.episode.to(device), t.step.to(device),
        )

    episodes = len(batch.target_kind)
    keys = machines[0].expand(episodes, 2, -1).clone()
    return lc.EpisodeTokens(gather(batch.support), gather(batch.query), gather(batch.chain), keys,
                            episodes, batch.target_kind.to(device))


@torch.no_grad()
def episode_predictions(core, tokens, loops=None):
    z = lc.codes_for(core, tokens.support, tokens.episodes, loops)
    q = tokens.query
    logit, _ = core.apply(q.m_pre, q.a, q.b, z[q.episode], loops)
    return logit


def nu_from_rows(rows, floors):
    """rows: dicts with family, group, truth, s, p. BA per group -> ν per family."""
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["group"]), []).append(r)
    per_family, skipped = {}, 0
    for (family, _), items in groups.items():
        truth = torch.tensor([r["truth"] for r in items])
        s = torch.tensor([r["s"] for r in items])
        predicted = torch.tensor([int(r["p"] >= 0.5) for r in items])
        if family in TRANSITION_FAMILIES:
            truth, predicted = truth ^ s, predicted ^ s
        ba = rw.balanced_accuracy(predicted, truth)
        if math.isnan(ba):
            skipped += 1
            continue
        floor = floors[family]["floor"]
        per_family.setdefault(family, []).append((ba - floor) / (1 - floor))
    result = {f: float(np.mean(v)) for f, v in per_family.items()}
    # A group whose queries contain only one scored class has undefined BA. It is
    # counted and makes the formal C2 screen incomplete; it is never a silent drop.
    result["groups_without_both_classes"] = skipped
    return result


def calibration(rows, bins=10):
    if not rows:
        return {}
    p = torch.tensor([r["p"] for r in rows])
    y = torch.tensor([float(r["truth"]) for r in rows])
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if mask.any():
            ece += float(mask.float().mean() * (p[mask].mean() - y[mask].mean()).abs())
    eps = 1e-6
    return dict(
        ece=ece,
        brier=float((p - y).square().mean()),
        nll=float(-(y * (p + eps).log() + (1 - y) * (1 - p + eps).log()).mean()),
        count=len(rows),
    )


def episode_metrics(core, perceive, batch, device, floors, loops=None):
    tokens = encode_episodes(perceive, batch, device)
    logit = episode_predictions(core, tokens, loops)
    rows = []
    for i in range(len(batch.query)):
        e = int(batch.query.episode[i])
        m = int(batch.query.machine[i])
        rows.append(
            dict(
                family=batch.rules[e].family,
                group=e,
                truth=int(batch.query.outcome[i]),
                s=int(batch.query.pre[i, m]),
                p=float(torch.sigmoid(logit[i])),
            )
        )
    return dict(nu=nu_from_rows(rows, floors), calibration=calibration(rows))


# ---------------------------------------------------------------- support-use controls (plan §23)

CONTROL_ARMS = ("full", "empty", "swapped", "permuted")


def permute_support_outcomes(support, generator):
    """Shuffle observed outcomes (m_post, outcome) among the support rows of each episode."""
    order = torch.arange(len(support))
    episode = support.episode.cpu()  # CPU generator and index arithmetic on any device
    for e in episode.unique().tolist():
        rows = (episode == e).nonzero().squeeze(1)
        order[rows] = rows[torch.randperm(len(rows), generator=generator)]
    order = order.to(support.m_post.device)
    return replace(support, m_post=support.m_post[order], outcome=support.outcome[order])


def swap_sources(rules):
    """Per episode, the next episode (cyclically) with a different rule; None if one has none."""
    sources = []
    for i, rule in enumerate(rules):
        others = [(i + k) % len(rules) for k in range(1, len(rules))]
        match = next((j for j in others if rules[j] != rule), None)
        if match is None:
            return None
        sources.append(match)
    return sources


def control_logits(core, tokens, arm, rules, *, seed=0, loops=None):
    """Query logits of one arm; query labels never enter. `rules` only choose swap partners."""
    support = tokens.support
    if arm == "permuted":
        support = permute_support_outcomes(support, torch.Generator().manual_seed(seed))
    if arm == "empty":
        width = support.m_pre.shape[-1]
        device = support.m_pre.device
        z = core.induce(torch.zeros(tokens.episodes, 0, width, device=device),
                        torch.zeros(tokens.episodes, 0, dtype=torch.bool, device=device), loops=loops)
    else:
        z = lc.codes_for(core, support, tokens.episodes, loops)
    if arm == "swapped":
        sources = swap_sources(rules)
        if sources is None:
            return None
        z = z[torch.tensor(sources, device=z.device)]
    elif arm not in ("full", "empty", "permuted"):
        raise ValueError(f"Unknown control arm {arm}")
    q = tokens.query
    return core.apply(q.m_pre, q.a, q.b, z[q.episode], loops)[0]


def control_metrics(core, perceive, batch, device, floors, *, seed=0, loops=None):
    """ν per family for every control arm on the same episodes; None where an arm is unavailable."""
    tokens = encode_episodes(perceive, batch, device)
    nu = {}
    for arm in CONTROL_ARMS:
        logit = control_logits(core, tokens, arm, batch.rules, seed=seed, loops=loops)
        if logit is None:
            nu[arm] = None
            continue
        rows = [dict(family=batch.rules[int(e)].family, group=int(e), truth=int(y), s=int(pre[m]),
                     p=float(torch.sigmoid(p.detach())))
                for e, y, pre, m, p in zip(batch.query.episode, batch.query.outcome, batch.query.pre,
                                           batch.query.machine, logit)]
        nu[arm] = nu_from_rows(rows, floors)
    return dict(nu=nu, episodes=len(batch.rules), rules=len(set(batch.rules)))


# ---------------------------------------------------------------- lives


@dataclass
class DemoSession:
    scene: rw.Scenes
    rules: tuple  # (left, right)
    labels: tuple  # evaluator kind labels per machine, e.g. ("A", "B")
    steps: list  # (lamps_pre [2], machine, i, j)


@dataclass
class QueryItem:
    scene: int
    lamps: tuple
    machine: int
    i: int
    j: int
    label: str
    truth: int  # evaluator only


@dataclass
class LifeSpec:
    population: str
    life_id: str
    kinds: dict
    rules: dict
    n_support: int
    acquisition: list
    corrupted: tuple  # (session, step)
    distraction: list
    query_scenes: list
    queries: list
    goals: list
    unavailable: dict
    counter: list
    claim: dict

    def poisoned(self):
        """Replace evaluator-only fields; the agent must behave identically."""
        queries = [replace(q, truth=1 - q.truth) for q in self.queries]
        strata = [g.stratum for g in self.goals]
        goals = [replace(g, stratum=strata[(i + 1) % len(strata)]) for i, g in enumerate(self.goals)]
        claim = dict(self.claim, truth=1 - self.claim["truth"])
        return replace(self, life_id="poisoned", queries=queries, goals=goals, claim=claim)


def _pair(g):
    return rw.PAIRS[int(torch.randint(len(rw.PAIRS), (1,), generator=g))]


def _scene(g, left_right_labels, kinds):
    labels = list(left_right_labels)
    if int(torch.randint(2, (1,), generator=g)):
        labels.reverse()
    scene = rw.sample_scenes(g, torch.tensor([[kinds[labels[0]], kinds[labels[1]]]]))
    return scene, tuple(labels)


def _sessions(g, labels, kinds, rules, per_machine, total, only=None):
    sessions = []
    remaining = total
    while remaining > 0:
        scene, order = _scene(g, labels, kinds)
        steps = []
        count = min(per_machine, remaining)
        for side, label in enumerate(order):
            if only is not None and label != only:
                continue
            for _ in range(count):
                i, j = _pair(g)
                steps.append((torch.randint(2, (2,), generator=g).tolist(), side, i, j))
        steps = [steps[k] for k in torch.randperm(len(steps), generator=g).tolist()]
        sessions.append(DemoSession(scene, tuple(rules[l] for l in order), order, steps))
        remaining -= count
    return sessions


def sample_life(generator, population, *, n_support, queries=64, goals=16, distract=32, counter=24, per_scene=8, cap=200, families=None):
    """One evaluation life. `families=(family_A, family_B)` fixes the queried rules'
    families from generator truth only (formal schedule); None samples uniformly
    from the population's rule pool (development)."""
    g = generator
    pool = list(rw.KIND_SPLIT[population])
    chosen = [pool[i] for i in torch.randperm(len(pool), generator=g)[:4].tolist()]
    kinds = dict(zip("ABCD", chosen))
    rule_pool = rw.split_rules()[population]

    def draw(candidates):
        return candidates[int(torch.randint(len(candidates), (1,), generator=g))]

    rules = {k: draw(rule_pool) for k in "ABCD"}
    if families is not None:
        for label, family in zip("AB", families):
            candidates = [r for r in rule_pool if r.family == family]
            if not candidates:
                raise ValueError(f"No {family} rule in the {population} pool")
            rules[label] = draw(candidates)
    acquisition = _sessions(g, ("A", "B"), kinds, rules, per_scene, n_support)
    a_steps = [(s, k) for s, session in enumerate(acquisition) for k, step in enumerate(session.steps) if session.labels[step[1]] == "A"]
    corrupted = a_steps[int(torch.randint(len(a_steps), (1,), generator=g))]
    distraction = _sessions(g, ("C", "D"), kinds, rules, per_scene, distract)
    seen = {"A": set(), "B": set()}
    for session in acquisition:
        attrs = session.scene.attrs[0].tolist()
        for lamps, side, i, j in session.steps:
            seen[session.labels[side]].add((tuple(attrs[i]), tuple(attrs[j]), lamps[side]))
    query_scenes = [_scene(g, ("A", "B"), kinds) for _ in range(4)]
    items = []
    for label in ("A", "B"):
        for q in range(queries):
            index = q % len(query_scenes)
            scene, order = query_scenes[index]
            side = order.index(label)
            attrs = scene.attrs[0].tolist()
            for _ in range(50):
                i, j = _pair(g)
                lamps = torch.randint(2, (2,), generator=g).tolist()
                if (tuple(attrs[i]), tuple(attrs[j]), lamps[side]) not in seen[label]:
                    break
            else:
                raise ValueError("Cannot draw a query disjoint from the acquisition inputs")
            truth = rw.outcome(rules[label], attrs[i], attrs[j], lamps[side])
            items.append(QueryItem(index, tuple(lamps), side, i, j, label, truth))
    tasks, unavailable = [], {}
    for stratum in rw.stratum_schedule(goals):
        order = ("A", "B") if int(torch.randint(2, (1,), generator=g)) else ("B", "A")
        try:
            tasks.append(
                rw.sample_task(g, torch.tensor([kinds[order[0]], kinds[order[1]]]), (rules[order[0]], rules[order[1]]), stratum, cap=cap)
            )
        except rw.StratumUnavailable:
            unavailable[stratum] = unavailable.get(stratum, 0) + 1
    counter_sessions = _sessions(g, ("A", "B"), kinds, rules, per_scene, counter, only="B")
    scene, order = _scene(g, ("A", "B"), kinds)
    side = order.index("A")
    i, j = _pair(g)
    lamps = torch.randint(2, (2,), generator=g).tolist()
    attrs = scene.attrs[0].tolist()
    truth = rw.outcome(rules["A"], attrs[i], attrs[j], lamps[side])
    # `claimed` is the observable value the agent receives (R1 claims are all false);
    # `truth` is an evaluator annotation that poisoning may flip.
    claim = dict(scene=scene, order=order, lamps=lamps, side=side, i=i, j=j, truth=truth, claimed=1 - truth)
    return LifeSpec(
        population, f"life-{int(torch.randint(10**9, (1,), generator=g))}", kinds, rules,
        n_support, acquisition, corrupted, distraction, [s for s, _ in query_scenes],
        items, tasks, unavailable, counter_sessions, claim,
    )


def record_for(scene, machine, i, j):
    m = scene.machine_xy[0, machine].tolist()
    return rw.ActionRecord(tuple(m), tuple(scene.object_xy[0, i].tolist()), tuple(scene.object_xy[0, j].tolist()))


def demo_transitions(session, contract, *, corrupt=None):
    """Harness-side demonstration: demonstrator sets lamps, presses, frames recorded."""
    world = rw.RuleWorld(session.scene, session.rules, torch.zeros(2, dtype=torch.long), rw.TaskContract(budget=10**9))
    transitions, clean = [], None
    for k, (lamps, side, i, j) in enumerate(session.steps):
        world.lamps = torch.tensor(lamps)
        pre = world.frame()
        record = record_for(session.scene, side, i, j)
        receipt = world.press(record)
        post = world.frame()
        if k == corrupt:
            clean = (pre, record, receipt.status, post)
            wrong = world.lamps.clone()
            wrong[side] = 1 - wrong[side]
            post = rw.render(session.scene, wrong[None])[0][0]
        transitions.append((pre, record, receipt.status, post))
    return transitions, clean


def fresh_agent(perception, core, directory, settings, device="cpu"):
    versions = dict(perception=state_hash(perception)[:16], core=state_hash(core)[:16])
    memory = ConceptMemory(directory, versions=versions)
    return ConceptAgent(perception, core, memory, rw.TaskContract(), settings, device=device)


def _true_label(session_labels, scene, xy):
    centers = scene.machine_xy[0]
    d = ((centers - torch.tensor(xy)) ** 2).sum(-1)
    return session_labels[int(d.argmin())]


@torch.no_grad()
def run_life(perception, core, spec, directory, *, settings, device="cpu", controls=True):
    if "claimed" not in spec.claim:
        raise ValueError("life spec claim lacks the observable 'claimed' value; refusing to derive it from truth")
    contract = rw.TaskContract()
    timings, hashes = {}, []
    outputs = dict(predictions=[], tasks=[], decisions=[])
    checks = {}

    def model_hash():
        return state_hash(perception) + state_hash(core)

    def clock(name, since):
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - since

    agent = fresh_agent(perception, core, directory / "memory", settings, device)
    hashes.append(model_hash())

    # 1-2. acquisition and distraction ------------------------------------
    start = time.perf_counter()
    bindings, corrupted_evidence, clean_transition = [], None, None
    evidence_frames = {}
    for phase, sessions in (("acquisition", spec.acquisition), ("distraction", spec.distraction)):
        for s, session in enumerate(sessions):
            corrupt = spec.corrupted[1] if phase == "acquisition" and s == spec.corrupted[0] else None
            transitions, clean = demo_transitions(session, contract, corrupt=corrupt)
            result = agent.observe_session(transitions)
            for k, e in enumerate(result["evidence"]):
                evidence_frames[e] = clean if k == corrupt else transitions[k]
            if corrupt is not None:
                corrupted_evidence, clean_transition = result["evidence"][corrupt], clean
            for d in result["decisions"]:
                bindings.append(dict(phase=phase, label=_true_label(session.labels, session.scene, d["xy"]), status=d["status"], concept=d["concept"], transitions=len(d["evidence"])))
    clock("acquire", start)
    hashes.append(model_hash())
    outputs["decisions"] = [(b["status"], b["concept"]) for b in bindings]

    founders = {}
    for b in bindings:
        if b["status"] == "created":
            founders.setdefault(b["concept"], b["label"])

    # queries --------------------------------------------------------------
    def frame_for(q):
        return rw.render(spec.query_scenes[q.scene], torch.tensor([q.lamps]))[0][0]

    def query_pass(tag, with_controls):
        rows, views = [], {}
        for q in spec.queries:
            key = (q.scene, q.lamps)
            if key not in views:
                views[key] = agent.observe_scene(frame_for(q), persist=False)
            view = views[key]
            record = record_for(spec.query_scenes[q.scene], q.machine, q.i, q.j)
            pred = agent.predict(view, record)
            rule = spec.rules[q.label]
            row = dict(
                tag=tag, label=q.label, family=rule.family, group=(spec.life_id, q.label),
                truth=q.truth, s=q.lamps[q.machine], p=pred["p"],
                retrieved=pred["concept"], retrieved_correct=founders.get(pred["concept"]) == q.label,
            )
            outputs["predictions"].append(pred["p"])
            if with_controls:
                owner = agent._owner(view.machines, record.machine_xy)
                other = view.machines[1 - owner] if len(view.machines) == 2 else None
                row["p_empty"] = agent.predict(view, record, code=agent.empty_code)["p"]
                if other is not None and other.get("concept") is not None:
                    row["p_swapped"] = agent.predict(view, record, code=agent.concept_state(other["concept"])[0])["p"]
                other_label = "B" if q.label == "A" else "A"
                other_rule = spec.rules[other_label]
                attrs = spec.query_scenes[q.scene].attrs[0].tolist()
                row["swap_truth"] = rw.outcome(other_rule, attrs[q.i], attrs[q.j], q.lamps[q.machine])
                oracle = next((c for c, label in founders.items() if label == q.label), None)
                if oracle is not None:
                    support = agent.support_of(oracle)
                    row["p_oracle_retrieval"] = agent.predict(view, record, code=agent.concept_state(oracle)[0])["p"]
                    if len(support) > 1:
                        permutation = torch.randperm(len(support), generator=torch.Generator().manual_seed(len(rows))).tolist()
                        row["p_permuted"] = agent.predict(view, record, code=agent.induce(support, permute=permutation))["p"]
                    for loops in (1, 4):
                        z = agent.induce(support, loops=loops)
                        row[f"p_loops{loops}"] = agent.predict(view, record, code=z, loops=loops)["p"]
            rows.append(row)
        return rows

    start = time.perf_counter()
    pre_restart = query_pass("pre_restart", False)
    clock("query", start)

    # 3. restart ------------------------------------------------------------
    agent.memory.save()
    memory_directory = agent.memory.directory
    versions = agent.memory.versions
    del agent
    start = time.perf_counter()
    memory = ConceptMemory.load(memory_directory, versions=versions)
    agent = ConceptAgent(perception, core, memory, contract, settings, device=device)
    clock("restart_load", start)
    hashes.append(model_hash())

    # 4. use ------------------------------------------------------------------
    start = time.perf_counter()
    use = query_pass("use", controls)
    clock("query", start)
    checks["restart_reproduces"] = all(
        abs(a["p"] - b["p"]) <= 1e-5 for a, b in zip(pre_restart, use)
    )
    start = time.perf_counter()
    tasks = []
    for t, task in enumerate(spec.goals):
        world = rw.RuleWorld(task.scene, task.rules, task.lamps.clone(), contract)
        out = agent.run_task(rw.Actuator(world), task.goal)
        tasks.append(_score_task(task, world, out, contract, "agent"))
        outputs["tasks"].append((out["option"], [s.get("action") for s in out["trace"]], out["presses"]))
        if controls:
            world = rw.RuleWorld(task.scene, task.rules, task.lamps.clone(), contract)
            out = agent.run_task(rw.Actuator(world), task.goal, control="empty")
            tasks.append(_score_task(task, world, out, contract, "empty_memory"))
            world = rw.RuleWorld(task.scene, task.rules, task.lamps.clone(), contract)
            tasks.append(_random_task(task, world, contract, t))
    clock("tasks", start)
    hashes.append(model_hash())

    # 5a. stale plan, then supersede ----------------------------------------
    start = time.perf_counter()
    b_rows_before = [r["p"] for r in use if r["label"] == "B"]
    task = spec.goals[0] if spec.goals else None
    if task is not None:
        world = rw.RuleWorld(task.scene, task.rules, task.lamps.clone(), contract)
        view = agent.observe_scene(world.frame(), persist=False)
        decision, read = agent.plan(view, task.goal, presses_done=0)
        if decision.option != "press":  # force an executable proposal for the check
            decision = lc.Decision("press", (0, 0, 1), 0.0, {}, 0)
    b_concepts = {r["retrieved"] for r in use if r["label"] == "B"} - {None}
    b_codes_before = {c: agent.concept_state(c) for c in b_concepts}
    new_evidence = agent.receive_supersede(corrupted_evidence, clean_transition)
    evidence_frames[new_evidence] = clean_transition
    if task is not None:
        if agent.memory.is_current(read):
            checks["stale_plan_rejected"] = None  # plan did not read the revised concept
        else:
            result = agent.execute(rw.Actuator(world), view, decision, read)
            checks["stale_plan_rejected"] = result["status"] == "stale" and world.presses == 0
    # Which concept now owns the replacement evidence?
    owner = next(
        (b.data["concept"] for b in agent._bindings()
         if b.data.get("supports") and new_evidence in agent.memory.view()["components"][b.parents[1]].evidence),
        None,
    )
    if owner is not None:
        z_after, _ = agent.concept_state(owner)
        support = agent.support_of(owner)
        fresh = ConceptAgent(perception, core, agent.memory, contract, settings, device=device)
        z_fresh = fresh.induce(support)
        checks["recompute_matches_active_support"] = bool(torch.allclose(z_after, z_fresh, atol=1e-5))
        def clean(e):
            # Harness-rendered clean demonstration frames; the agent's own executed
            # observations (not harness demonstrations) are used verbatim from blobs.
            if e in evidence_frames:
                pre, record, _, post = evidence_frames[e]
                return fresh.transition_tokens(pre, record, post)["e"]
            record = agent.memory.view()["evidence"][e]
            frames = agent.memory.get_blob(record.content_ref, record.content_hash)
            action = record.data["action"]
            return fresh.transition_tokens(
                torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 255,
                rw.ActionRecord(tuple(action["machine_xy"]), tuple(action["a_xy"]), tuple(action["b_xy"])),
                torch.from_numpy(frames[1]).permute(2, 0, 1).float() / 255,
            )["e"]

        clean_tokens = [clean(e) for e in support]
        z_clean = core.induce(torch.stack(clean_tokens)[None], torch.ones(1, len(support), dtype=torch.bool, device=z_after.device), loops=settings.loops)[0]
        checks["supersede_matches_clean"] = bool(torch.allclose(z_after, z_clean, atol=1e-5))
    else:
        checks["recompute_matches_active_support"] = None
        checks["supersede_matches_clean"] = None
    after_supersede = query_pass("after_supersede", False)
    b_rows_after = [r["p"] for r in after_supersede if r["label"] == "B"]
    unaffected = [c for c in b_concepts if c != owner]
    checks["unaffected_bit_identical"] = bool(
        all(torch.equal(b_codes_before[c][0], agent.concept_state(c)[0]) and b_codes_before[c][1] == agent.concept_state(c)[1] for c in unaffected)
        and (b_rows_before == b_rows_after if owner not in b_concepts else True)
    )
    clock("supersede", start)

    # 5b. counter evidence for B -------------------------------------------
    start = time.perf_counter()
    for session in spec.counter:
        transitions, _ = demo_transitions(session, contract)
        result = agent.observe_session(transitions)
        for d in result["decisions"]:
            bindings.append(dict(phase="counter", label=_true_label(session.labels, session.scene, d["xy"]), status=d["status"], concept=d["concept"], transitions=len(d["evidence"])))
    after_counter = query_pass("after_counter", False)
    a_before = [r["p"] for r in after_supersede if r["label"] == "A"]
    a_after = [r["p"] for r in after_counter if r["label"] == "A"]
    behaviour = dict(a_answers_unchanged_by_b_counter_evidence=a_before == a_after)
    clock("counter", start)

    # 5c. testimony ------------------------------------------------------------
    start = time.perf_counter()
    claim = spec.claim
    world = rw.RuleWorld(claim["scene"], tuple(spec.rules[l] for l in claim["order"]), torch.tensor(claim["lamps"]), contract)
    frame = world.frame()
    record = record_for(claim["scene"], claim["side"], claim["i"], claim["j"])
    tested = agent.receive_claim(frame, record, claim["claimed"], rw.Actuator(world))
    episodic = agent.episodic_answer(frame, record)
    view = agent.observe_scene(frame, persist=False)
    concept_p = agent.predict(view, record)["p"]
    final = episodic["outcome"] if episodic is not None else float(concept_p >= 0.5)
    environment_truth = int(world.lamps[claim["side"]])
    observed = tested["observed"]
    outputs["claim"] = dict(transmitted=claim["claimed"], tested=tested["tested"], observed=observed, concept_p=concept_p, final=final)
    claim_result = dict(
        tested=tested["tested"], observed=observed, final=final,
        # Exact recall of the agent's own just-observed test: episodic, not learned.
        episodic_recall_follows_observation=None if observed is None else final == observed,
        # Concept-level answer after the test feedback entered the acquisition path.
        concept_level_follows_observation=None if observed is None else int(concept_p >= 0.5) == observed,
        concept_level_correct=int(concept_p >= 0.5) == claim["truth"],
        final_matches_environment=final == environment_truth,
        feedback=None if tested["feedback"] is None else tested["feedback"]["status"],
        claim_supported=False,
    )
    clock("claim", start)
    hashes.append(model_hash())
    checks["weights_unchanged"] = len(set(hashes)) == 1
    agent.memory.save()  # measure AFTER publishing the final durable state
    disk = agent.memory.disk_bytes()
    return dict(
        life=spec.life_id, n_support=spec.n_support, population=spec.population,
        rules={k: v.key() for k, v in spec.rules.items()},
        rule_families={k: v.family for k, v in spec.rules.items()},
        queries=pre_restart + use + after_supersede + after_counter,
        tasks=tasks, bindings=bindings, claim=claim_result, checks=checks, behaviour=behaviour,
        hashes=hashes, agent_outputs=outputs, timings=timings,
        unavailable=spec.unavailable,
        memory=dict(**disk, events=agent.memory.store.revision, caches=agent.cache_stats()),
    )


def _score_task(task, world, out, contract, arm):
    if out["option"] == "stop":
        outcome = "success" if world.verify(task.goal) else "false_stop"
    else:
        outcome = "abstain"
    rejected = sum(r.status == "budget_exceeded" for r in world.receipts)
    feedback = [s.get("feedback") for s in out["trace"] if s.get("feedback")]
    return dict(
        arm=arm, stratum=task.stratum, option=out["option"], outcome=outcome,
        presses=world.presses, rejected=rejected,
        utility=contract.utility(outcome, world.presses),
        stale=sum(s.get("status") == "stale" for s in out["trace"]),
        feedback_support=sum(f["status"] in ("bound", "created", "appearance_support") for f in feedback),
    )


def _random_task(task, world, contract, index):
    g = torch.Generator().manual_seed(10007 + index)
    for _ in range(int(torch.randint(contract.budget + 1, (1,), generator=g))):
        m = int(torch.randint(2, (1,), generator=g))
        i, j = _pair(g)
        world.press(record_for(task.scene, m, i, j))
    outcome = "success" if world.verify(task.goal) else "false_stop"
    return dict(arm="random", stratum=task.stratum, option="stop", outcome=outcome,
                presses=world.presses, rejected=0, utility=contract.utility(outcome, world.presses), stale=0)


# ---------------------------------------------------------------- perception metrics


@torch.no_grad()
def perception_metrics(perception, generator, population, count, device, chunk=64):
    kinds = torch.tensor(rw.KIND_SPLIT[population])
    totals = dict(attributes=torch.zeros(4), lamp=0.0, machine_pointer=0.0, object_pointer=0.0, machine_detection=0.0, reconstruction=0.0)
    seen = 0
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        pick = kinds[torch.randint(len(kinds), (n, 2), generator=generator)]
        scenes = rw.sample_scenes(generator, pick)
        lamps = torch.randint(2, (n, 2), generator=generator)
        rgb, entity = rw.render(scenes, lamps)
        p = perception(rgb.to(device))
        entity = entity.to(device)
        owner = p.alpha.argmax(1)  # [B,H,W] winning slot
        rows = torch.arange(n, device=device)

        def dominant(slot):
            # entity with most pixels among those the slot wins
            counts = torch.stack([((owner == slot[:, None, None]) & (entity == e)).flatten(1).sum(-1) for e in range(7)], -1)
            return counts.argmax(-1)

        for m in range(2):
            slot = pointer(p.alpha, scenes.machine_xy[:, m].to(device))
            totals["machine_pointer"] += float((dominant(slot) == 1 + m).float().sum())
            totals["lamp"] += float(((p.lamp[rows, slot] > 0).long().cpu() == lamps[:, m]).float().sum())
        for i in range(4):
            slot = pointer(p.alpha, scenes.object_xy[:, i].to(device))
            totals["object_pointer"] += float((dominant(slot) == 3 + i).float().sum())
            predicted = p.attributes[rows, slot].argmax(-1).cpu()
            totals["attributes"] += (predicted == scenes.attrs[:, i]).float().sum(0)
        machine_slots = p.kind.softmax(-1)[..., 1].topk(2, dim=-1).indices
        found = torch.stack([dominant(machine_slots[:, k]) for k in range(2)], -1).sort(-1).values
        totals["machine_detection"] += float((found == torch.tensor([1, 2], device=device)).all(-1).float().sum())
        totals["reconstruction"] += float((p.recon - rgb.to(device)).square().mean((1, 2, 3)).sum())
        seen += n
    return dict(
        attribute_accuracy=(totals["attributes"] / (4 * seen)).tolist(),
        lamp_accuracy=totals["lamp"] / (2 * seen),
        machine_pointer_accuracy=totals["machine_pointer"] / (2 * seen),
        object_pointer_accuracy=totals["object_pointer"] / (4 * seen),
        machine_detection=totals["machine_detection"] / seen,
        reconstruction_mse=totals["reconstruction"] / seen,
        scenes=seen,
    )


# ---------------------------------------------------------------- summaries and gates

THRESHOLDS = dict(
    C1_attribute=0.95, C1_lamp=0.99, C1_machine_pointer=0.99, C1_object_pointer=0.97,
    C2_nu=0.7, C2_ece=0.05, C3_revisit=0.90, C3_new=0.90, C3_false_merge=0.10,
    C4_restart_margin=-0.03, C4_reach=0.85, C4_already=0.90, C4_unreachable_abstain=0.85,
    C4_false_stop=0.10, C5c=0.95,
)
FORMAL_N = 128  # C2/C4 are defined at N*=128, never at "the largest N that ran"
COUNTER_N = 8  # C5b: counter-evidence after acquisition with N0=8
FORMAL_SEEDS = 3
FORMAL_PROTOCOL = dict(
    population="test", size="full", model_size="full", life_supports=[8, 32, 128],
    lives_per_support=4, life_queries=64, life_goals=16, distract=32, counter=24,
    family_schedule=[list(pair) for pair in FAMILY_SCHEDULE],
)
FAMILY_ORDER = ("category", "relation") + TRANSITION_FAMILIES
REQUIRED_CHECKS = (
    "restart_reproduces", "stale_plan_rejected", "recompute_matches_active_support",
    "supersede_matches_clean", "unaffected_bit_identical", "weights_unchanged",
)
PASS, FAIL, INCOMPLETE, INELIGIBLE = "pass", "fail", "incomplete", "ineligible"


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def _status(value):
    return INCOMPLETE if value is None else (PASS if value else FAIL)


def summarize(lives, floors, perception=None, provenance=None):
    """Point estimates for ONE evaluation (one training seed) plus a screen.

    Missing or ineligible measurements stay None; they are never counted as passes.
    """
    out = dict(lives=len(lives), provenance=provenance, perception=perception)
    by_n = {}
    for life in lives:
        by_n.setdefault(life["n_support"], []).append(life)
    out["lives_by_support"] = {n: len(v) for n, v in sorted(by_n.items())}

    def rows(tag, n=None, label=None):
        return [
            r for l in lives if n is None or l["n_support"] == n
            for r in l["queries"] if r["tag"] == tag and (label is None or r["label"] == label)
        ]

    out["nu_by_support"] = {n: nu_from_rows(rows("use", n), floors) for n in sorted(by_n)}
    formal = rows("use", FORMAL_N)
    out["nu_full"] = nu_from_rows(formal, floors)
    out["family_coverage"] = {
        n: sorted({r["family"] for r in rows("use", n)}) for n in sorted(by_n)
    }
    for arm in ("p_empty", "p_permuted", "p_oracle_retrieval", "p_loops1", "p_loops4"):
        out[f"nu_{arm[2:]}"] = nu_from_rows([dict(r, p=r[arm]) for r in formal if arm in r], floors)
    out["nu_pre_restart_formal"] = nu_from_rows(rows("pre_restart", FORMAL_N), floors)
    informative = [r for r in formal if "p_swapped" in r and r["swap_truth"] != r["truth"]]
    out["swap"] = dict(
        informative=len(informative),
        coverage=len(informative) / len(formal) if formal else None,
        follows_swapped_rule_under_swap=_mean(int(r["p_swapped"] >= 0.5) == r["swap_truth"] for r in informative),
        follows_swapped_rule_own_code=_mean(int(r["p"] >= 0.5) == r["swap_truth"] for r in informative),
    )
    out["calibration"] = calibration(formal)  # finite-population evidence at N=128
    out["retrieval_correct_use"] = _mean(r["retrieved_correct"] for r in rows("use"))
    revisit, new, merges = [], [], []
    for life in lives:
        seen_labels, owners = set(), {}
        for b in life["bindings"]:
            if b["transitions"] < 4:  # behavioural verification needs m>=4
                continue
            if b["status"] == "created":
                owners[b["concept"]] = b["label"]
            if b["label"] not in seen_labels:
                new.append(b["status"] == "created")
                seen_labels.add(b["label"])
            else:
                bound = b["status"] in ("bound", "repaired")
                revisit.append(bound and owners.get(b["concept"]) == b["label"])
                merges.append(bound and owners.get(b["concept"]) not in (None, b["label"]))
    out["bindings"] = dict(
        revisit_correct=_mean(revisit), new_created=_mean(new), false_merge=_mean(merges),
        revisits=len(revisit), first_sightings=len(new),
    )
    formal_tasks = [t for l in by_n.get(FORMAL_N, []) for t in l["tasks"]]
    out["unavailable_strata_formal"] = {}
    for life in by_n.get(FORMAL_N, []):
        for k, v in life["unavailable"].items():
            out["unavailable_strata_formal"][k] = out["unavailable_strata_formal"].get(k, 0) + v

    def rate(arm, strata, value):
        chosen = [t for t in formal_tasks if t["arm"] == arm and t["stratum"] in strata]
        return dict(rate=_mean(t["outcome"] == value for t in chosen), count=len(chosen))

    out["tasks"] = {}
    for arm in ("agent", "empty_memory", "random"):
        chosen = [t for t in formal_tasks if t["arm"] == arm]
        out["tasks"][arm] = dict(
            reach=rate(arm, ("reach1", "reach2"), "success"),
            reach1=rate(arm, ("reach1",), "success"),
            reach2=rate(arm, ("reach2",), "success"),
            already=rate(arm, ("already",), "success"),
            unreachable_abstain=rate(arm, ("unreachable",), "abstain"),
            false_stop=dict(rate=_mean(t["outcome"] == "false_stop" for t in chosen), count=len(chosen)),
            mean_utility=_mean(t["utility"] for t in chosen),
            rejected=sum(t["rejected"] for t in chosen),
            stale=sum(t["stale"] for t in chosen),
            feedback=sum(t.get("feedback_support", 0) for t in chosen),
        )
    before = nu_from_rows(rows("after_supersede", COUNTER_N, "B"), floors)
    after = nu_from_rows(rows("after_counter", COUNTER_N, "B"), floors)
    shared = [f for f in FAMILY_ORDER if f in before and f in after]
    out["counter_evidence"] = dict(
        n_support=COUNTER_N, before=before, after=after,
        gain=_mean(after[f] - before[f] for f in shared) if shared else None,
    )
    claims = [l["claim"] for l in lives]
    out["claims"] = {
        k: _mean(c[k] for c in claims if c[k] is not None)
        for k in ("tested", "episodic_recall_follows_observation",
                  "concept_level_follows_observation", "concept_level_correct")
    }
    for name in ("checks", "behaviour"):
        collected = {}
        for life in lives:
            for k, v in life[name].items():
                collected.setdefault(k, []).append(v)
        out[name] = {
            k: dict(true=sum(v is True for v in vs), false=sum(v is False for v in vs),
                    not_applicable=sum(v is None for v in vs))
            for k, vs in collected.items()
        }
    out["screen"] = screen(out)
    return out


def _at_least(value, threshold):
    return None if value is None or math.isnan(value) else value >= threshold


def _at_most(value, threshold):
    return None if value is None or math.isnan(value) else value <= threshold


def screen(s):
    """Single-seed point-estimate screen: pass / fail / incomplete per gate.

    Formal decisions use `aggregate_gates` over three training seeds.
    """
    t = THRESHOLDS
    result = {}
    p = s.get("perception")
    result["C1"] = INCOMPLETE if not p else _status(
        min(p["attribute_accuracy"]) >= t["C1_attribute"] and p["lamp_accuracy"] >= t["C1_lamp"]
        and p["machine_pointer_accuracy"] >= t["C1_machine_pointer"]
        and p["object_pointer_accuracy"] >= t["C1_object_pointer"]
    )
    nu, empty = s["nu_full"], s["nu_empty"]
    values = [nu.get(f) for f in FAMILY_ORDER]
    undefined = nu.get("groups_without_both_classes", 0) > 0
    result["C2_nu"] = INCOMPLETE if None in values or undefined else _status(all(v >= t["C2_nu"] for v in values))
    gains = [None if f not in nu or f not in empty else nu[f] - empty[f] for f in FAMILY_ORDER]
    result["C2_memory_gain"] = INCOMPLETE if None in gains else _status(all(g > 0 for g in gains))
    sw = s["swap"]
    under, own = sw["follows_swapped_rule_under_swap"], sw["follows_swapped_rule_own_code"]
    result["C2_swap_shift"] = INCOMPLETE if under is None or own is None else _status(under > own)
    result["C2_ece"] = _status(_at_most(s["calibration"].get("ece"), t["C2_ece"]))
    b = s["bindings"]
    parts = [_at_least(b["revisit_correct"], t["C3_revisit"]), _at_least(b["new_created"], t["C3_new"]),
             _at_most(b["false_merge"], t["C3_false_merge"])]
    result["C3"] = INCOMPLETE if None in parts else _status(all(parts))
    a, e = s["tasks"]["agent"], s["tasks"]["empty_memory"]
    parts = [
        _at_least(a["reach"]["rate"], t["C4_reach"]), _at_least(a["already"]["rate"], t["C4_already"]),
        _at_least(a["unreachable_abstain"]["rate"], t["C4_unreachable_abstain"]),
        _at_most(a["false_stop"]["rate"], t["C4_false_stop"]),
        None if a["mean_utility"] is None or e["mean_utility"] is None else a["mean_utility"] > e["mean_utility"],
    ]
    missing_strata = any(a[k]["count"] == 0 for k in ("reach1", "reach2", "already", "unreachable_abstain"))
    if s["unavailable_strata_formal"] or missing_strata or None in parts:
        result["C4"] = INCOMPLETE
    else:
        result["C4"] = _status(all(parts))
    gain = s["counter_evidence"]["gain"]
    result["C5b"] = INCOMPLETE if gain is None else _status(gain > 0)
    c = s["claims"]
    parts = [_at_least(c["tested"], t["C5c"]), _at_least(c["episodic_recall_follows_observation"], t["C5c"])]
    result["C5c"] = INCOMPLETE if None in parts else _status(all(parts))
    checks = s["checks"]
    if any(checks.get(k, {}).get("false", 0) for k in REQUIRED_CHECKS):
        result["I_runtime_checks"] = FAIL
    elif all(k in checks and checks[k]["true"] == s["lives"] for k in REQUIRED_CHECKS):
        result["I_runtime_checks"] = PASS
    else:  # a check that did not run (n/a) is unestablished, not passed
        result["I_runtime_checks"] = INCOMPLETE
    return result


def lcb(values):
    """One-sided 95% t lower bound; any missing seed measurement makes it undefined."""
    values = list(values)
    if len(values) < 2 or any(v is None or not math.isfinite(v) for v in values):
        return None
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return mean - T_95[len(values) - 1] * sd / math.sqrt(len(values))


def formal_eligibility(summaries):
    """Reasons why these evaluations are not a formal R1 decision (empty = eligible)."""
    reasons = []
    if len(summaries) != FORMAL_SEEDS:
        reasons.append(f"need exactly {FORMAL_SEEDS} evaluations, got {len(summaries)}")
    provenance = [s.get("provenance") for s in summaries]
    if any(p is None for p in provenance):
        return reasons + ["missing training/evaluation provenance"]
    for key in ("core_seed", "perception_seed", "core_sha256", "perception_sha256"):
        values = [p.get(key) for p in provenance]
        if None in values or len(set(values)) != len(values):
            reasons.append(f"{key} must be declared and distinct per training seed")
    for key in ("protocol", "evaluation_seed", "source_sha256", "floors_sha256"):
        values = [json_key(p.get(key)) for p in provenance]
        if None in [p.get(key) for p in provenance] or len(set(values)) != 1:
            reasons.append(f"{key} must be declared and identical across seeds")
    if any(p.get("protocol") != FORMAL_PROTOCOL for p in provenance):
        reasons.append("protocol differs from the full formal R1 protocol (test population, full-size models, N=8/32/128, 4 lives)")
    for s in summaries:
        counts = {int(k): v for k, v in s.get("lives_by_support", {}).items()}  # JSON keys are strings
        if counts.get(FORMAL_N) != FORMAL_PROTOCOL["lives_per_support"]:
            reasons.append("each evaluation needs the declared N=128 lives")
            break
    return reasons


def json_key(value):
    import json

    return json.dumps(value, sort_keys=True, default=str)


def aggregate_gates(summaries):
    """Formal gates over the three training seeds (one-sided 95% t lower bounds).

    Ineligible provenance/protocol or any missing measurement never passes.
    """
    t = THRESHOLDS
    names = ("C1", "C2", "C3", "C4", "C4_restart", "C4_utility_gain", "C5b", "C5c", "I_runtime_checks")
    reasons = formal_eligibility(summaries)
    if reasons:
        return dict(status=INELIGIBLE, reasons=reasons, seeds=len(summaries), bounds={},
                    gates=dict.fromkeys(names, INELIGIBLE))

    def difference(a, b):
        return None if a is None or b is None else a - b

    bounds = {}
    for f in FAMILY_ORDER:
        bounds[f"nu_{f}"] = lcb([s["nu_full"].get(f) for s in summaries])
        bounds[f"gain_{f}"] = lcb([difference(s["nu_full"].get(f), s["nu_empty"].get(f)) for s in summaries])
    bounds["swap_shift"] = lcb([difference(s["swap"]["follows_swapped_rule_under_swap"], s["swap"]["follows_swapped_rule_own_code"]) for s in summaries])
    bounds["restart"] = lcb([
        None if any(f not in s["nu_full"] or f not in s["nu_pre_restart_formal"] for f in FAMILY_ORDER)
        else float(np.mean([s["nu_full"][f] - s["nu_pre_restart_formal"][f] for f in FAMILY_ORDER]))
        for s in summaries
    ])
    bounds["utility_gain"] = lcb([difference(s["tasks"]["agent"]["mean_utility"], s["tasks"]["empty_memory"]["mean_utility"]) for s in summaries])
    bounds["counter_gain"] = lcb([s["counter_evidence"]["gain"] for s in summaries])

    def bound_gate(value, predicate):
        return INCOMPLETE if value is None else _status(predicate(value))

    c2_parts = [bound_gate(bounds[f"nu_{f}"], lambda v: v >= t["C2_nu"]) for f in FAMILY_ORDER]
    c2_parts += [bound_gate(bounds[f"gain_{f}"], lambda v: v > 0) for f in FAMILY_ORDER]
    c2_parts += [bound_gate(bounds["swap_shift"], lambda v: v > 0)]
    c2_parts += [s["screen"]["C2_ece"] for s in summaries]
    c2_parts += [INCOMPLETE if s["nu_full"].get("groups_without_both_classes", 0) else PASS for s in summaries]
    gates = dict(C2=_combine(c2_parts))
    gates["C4_restart"] = bound_gate(bounds["restart"], lambda v: v >= t["C4_restart_margin"])
    gates["C4_utility_gain"] = bound_gate(bounds["utility_gain"], lambda v: v > 0)
    gates["C5b"] = bound_gate(bounds["counter_gain"], lambda v: v > 0)
    for key in ("C1", "C3", "C4", "C5c", "I_runtime_checks"):
        gates[key] = _combine([s["screen"].get(key, INCOMPLETE) for s in summaries])
    return dict(status=_combine(gates.values()), seeds=len(summaries), bounds=bounds, gates=gates)


def _combine(statuses):
    statuses = list(statuses)
    if any(s == FAIL for s in statuses):
        return FAIL
    if all(s == PASS for s in statuses):
        return PASS
    return INCOMPLETE
