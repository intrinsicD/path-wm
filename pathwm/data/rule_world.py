"""RuleWorld-64: two independently controlled rule machines and four static objects.

Hidden rules come from the 280-rule grammar of the shared-abstraction experiment,
restricted to executed presses (u=1). The environment owns pixels, exact entity
masks, rules and lamps. An agent receives only frames, action receipts and goals;
masks are training labels and executor resolution, never agent inputs.
"""

from dataclasses import dataclass, field
from itertools import combinations
import hashlib
import json
import math

import torch

SIZE = 64
FAMILIES = ("category", "relation", "open", "close", "toggle")
FAMILY_INDEX = {f: i for i, f in enumerate(FAMILIES)}
SPLIT_SALT = "shared-abstraction-v1"
KIND_COUNT = 72
KIND_SPLIT = {
    "train": tuple(range(0, 48)),
    "validation": tuple(range(48, 56)),
    "test": tuple(range(56, 72)),
}
STRATA = {"already": 0.15, "reach1": 0.35, "reach2": 0.25, "unreachable": 0.25}
ENTITIES = 7  # background, left machine, right machine, four objects
BACKGROUND = (0.32, 0.34, 0.38)
PANEL = (0.72, 0.72, 0.72)
LAMP = {0: (0.12, 0.12, 0.14), 1: (1.0, 0.93, 0.35)}
PALETTE = torch.tensor(
    [[0.85, 0.15, 0.15], [0.20, 0.75, 0.20], [0.20, 0.30, 0.90], [0.90, 0.85, 0.20]]
)


# ------------------------------------------------------------------ rules and split


@dataclass(frozen=True)
class Rule:
    family: str
    j: int
    k: int = 0
    delta: int = 0
    values: tuple[int, ...] = ()

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("Unknown rule family")

    def key(self):
        return json.dumps(
            [self.family, self.j, self.k, self.delta, list(self.values)],
            separators=(",", ":"),
        )

    def group(self):
        """Split group: a category with its complement; one matching predicate."""
        if self.family == "category":
            other = [v for v in range(4) if v not in self.values]
            return ("category", self.j, tuple(min(list(self.values), other)))
        return ("predicate", self.j, self.k, self.delta)


def grammar():
    rules = [
        Rule("category", j, values=vs)
        for j in range(4)
        for vs in combinations(range(4), 2)
    ]
    rules += [
        Rule("relation", j, k, d) for j in range(4) for k in range(4) for d in range(4)
    ]
    rules += [
        Rule(op, j, k, d)
        for j in range(4)
        for k in range(4)
        for d in range(4)
        for op in ("open", "close", "toggle")
    ]
    return tuple(rules)


def outcome(rule, a, b, s):
    """Scalar truth for one executed press (u=1)."""
    if rule.family == "category":
        return int(a[rule.j] in rule.values)
    matched = (a[rule.j] + rule.delta) % 4 == b[rule.k]
    if rule.family == "relation":
        return int(matched)
    if rule.family == "open":
        return int(bool(s) or matched)
    if rule.family == "close":
        return int(bool(s) and not matched)
    return int(bool(s) != matched)


def rule_tensors(rules, device=None):
    vmask = torch.zeros(len(rules), 4, dtype=torch.bool)
    for i, r in enumerate(rules):
        vmask[i, list(r.values)] = True
    result = dict(
        family=torch.tensor([FAMILY_INDEX[r.family] for r in rules]),
        j=torch.tensor([r.j for r in rules]),
        k=torch.tensor([r.k for r in rules]),
        delta=torch.tensor([r.delta for r in rules]),
        vmask=vmask,
    )
    return {k: v.to(device) for k, v in result.items()} if device else result


def outcome_batch(t, a, b, s):
    """Vectorized truth; t holds per-row rule tensors, a/b [N,4], s [N]."""
    aj = a.gather(1, t["j"][:, None]).squeeze(1)
    bk = b.gather(1, t["k"][:, None]).squeeze(1)
    m = (aj + t["delta"]) % 4 == bk
    cat = t["vmask"].gather(1, aj[:, None]).squeeze(1)
    s = s.bool()
    f = t["family"]
    out = torch.where(
        f == 0,
        cat,
        torch.where(
            f == 1, m, torch.where(f == 2, s | m, torch.where(f == 3, s & ~m, s ^ m))
        ),
    )
    return out.long()


def _ordered(groups):
    return sorted(
        groups,
        key=lambda g: hashlib.sha256(
            (SPLIT_SALT + json.dumps(g, separators=(",", ":"))).encode()
        ).hexdigest(),
    )


def split_groups():
    """Deterministic group split (same construction as the approved proposal)."""
    cat = _ordered(
        [
            [j, list(vs)]
            for j in range(4)
            for vs in combinations(range(4), 2)
            if list(vs) < [a for a in range(4) if a not in vs]
        ]
    )
    rel = _ordered([[j, k, d] for j in range(4) for k in range(4) for d in range(4)])
    return {
        "category": dict(train=cat[:8], validation=cat[8:10], test=cat[10:]),
        "predicate": dict(train=rel[:44], validation=rel[44:54], test=rel[54:]),
    }


def split_rules():
    groups = split_groups()
    lookup = {}
    for part, items in groups["category"].items():
        for j, vs in items:
            lookup[("category", j, tuple(vs))] = part
    for part, items in groups["predicate"].items():
        for j, k, d in items:
            lookup[("predicate", j, k, d)] = part
    result = {"train": [], "validation": [], "test": []}
    for rule in grammar():
        result[lookup[rule.group()]].append(rule)
    return {k: tuple(v) for k, v in result.items()}


# ------------------------------------------------------------------ scenes and rendering


def _kind_table():
    colors, patterns, periods = [], [], []
    for kind in range(KIND_COUNT):
        g = torch.Generator().manual_seed(7919 + kind)
        while True:
            pair = 0.2 + 0.75 * torch.rand(2, 3, generator=g)
            if (pair[0] - pair[1]).abs().mean() > 0.25:
                break
        colors.append(pair)
        patterns.append(kind % 6)
        periods.append(2 + (kind // 6) % 3)
    return torch.stack(colors), torch.tensor(patterns), torch.tensor(periods)


KIND_COLORS, KIND_PATTERN, KIND_PERIOD = _kind_table()
HELDOUT_KINDS = KIND_SPLIT["validation"] + KIND_SPLIT["test"]


@dataclass(frozen=True)
class Textures:
    """Explicit machine body textures per scene: colours [B,2,2,3], pattern/period [B,2].

    Rendering-only nuisance; lamp, panel, geometry, masks and labels do not depend on it.
    """

    colors: torch.Tensor
    pattern: torch.Tensor
    period: torch.Tensor

    def where(self, mask, other):
        """Per-machine choice [B,2]: `other` where mask is True, else self."""
        return Textures(
            torch.where(mask[..., None, None], other.colors, self.colors),
            torch.where(mask, other.pattern, self.pattern),
            torch.where(mask, other.period, self.period),
        )


def kind_textures(kinds):
    """The fixed texture table for kind ids [B,2] (the default renderer's lookup)."""
    kinds = torch.as_tensor(kinds).long()
    return Textures(KIND_COLORS[kinds], KIND_PATTERN[kinds], KIND_PERIOD[kinds])


@dataclass(frozen=True)
class TextureSampler:
    """S1-only procedural body textures (declared training-data design).

    Colour pairs, pattern and period follow the kind generator's ranges; with
    probability `near_lamp` one colour is drawn within `radius` (per channel) of the
    rendered lamp-on or lamp-off colour. Any texture whose pattern and period equal a
    declared held-out (validation/test) texture and whose two colours are both within
    `exclusion` (Euclidean) of that texture's colours is rejected. This partition is
    fixed by kind ids, never by model outcomes. At most `cap` draws per texture.
    """

    near_lamp: float = 0.25
    radius: float = 0.12
    exclusion: float = 0.1
    cap: int = 64
    heldout: tuple = HELDOUT_KINDS

    def heldout_like(self, colors, pattern, period):
        kinds = torch.tensor(self.heldout)
        same = (KIND_PATTERN[kinds] == pattern) & (KIND_PERIOD[kinds] == period)
        close = (KIND_COLORS[kinds] - colors).norm(dim=-1).le(self.exclusion).all(-1)
        return bool((same & close).any())

    def sample(self, generator, count):
        """-> Textures [count,2] and draw statistics (deterministic in `generator`)."""
        g = generator
        colors = torch.empty(count, 2, 2, 3)
        pattern = torch.empty(count, 2, dtype=torch.long)
        period = torch.empty(count, 2, dtype=torch.long)
        stats = dict(textures=0, draws=0, heldout_rejections=0, near_lamp=0)
        anchors = torch.tensor([LAMP[1], LAMP[0]])
        for b in range(count):
            for m in range(2):
                # Decide near-lamp once per texture: retries must not bias its rate.
                near = float(torch.rand(1, generator=g)) < self.near_lamp
                anchor = anchors[int(torch.randint(2, (1,), generator=g))]
                index = int(torch.randint(2, (1,), generator=g))
                for _ in range(self.cap):
                    stats["draws"] += 1
                    pair = 0.2 + 0.75 * torch.rand(2, 3, generator=g)
                    if near:
                        jitter = (torch.rand(3, generator=g) * 2 - 1) * self.radius
                        pair[index] = (anchor + jitter).clamp(0, 1)
                    kind_pattern = int(torch.randint(6, (1,), generator=g))
                    kind_period = int(torch.randint(2, 5, (1,), generator=g))
                    if (pair[0] - pair[1]).abs().mean() <= 0.25:
                        continue
                    if self.heldout_like(pair, kind_pattern, kind_period):
                        stats["heldout_rejections"] += 1
                        continue
                    break
                else:
                    raise ValueError(f"Texture sampler exhausted its cap of {self.cap} draws")
                colors[b, m], pattern[b, m], period[b, m] = pair, kind_pattern, kind_period
                stats["textures"] += 1
                stats["near_lamp"] += int(near)
        return Textures(colors, pattern, period), stats


@dataclass(frozen=True)
class Scenes:
    """Hidden scene structure. The agent never receives this object."""

    kind: torch.Tensor  # [B,2] machine texture ids (left, right)
    machine_xy: torch.Tensor  # [B,2,2] pixel centers (x, y)
    attrs: torch.Tensor  # [B,4,4] color, shape, size, pattern
    object_xy: torch.Tensor  # [B,4,2]

    def __len__(self):
        return len(self.kind)

    def select(self, index):
        return Scenes(*(v[index] for v in vars(self).values()))

    @staticmethod
    def cat(parts):
        return Scenes(*(torch.cat([vars(p)[n] for p in parts]) for n in vars(parts[0])))


def sample_scenes(generator, kinds, attrs=None):
    kinds = torch.as_tensor(kinds).long().reshape(-1, 2)
    b = len(kinds)
    g = generator

    def jitter(low, high, *shape):
        return torch.randint(low, high + 1, shape, generator=g).float()

    machine_xy = torch.stack(
        (
            torch.stack((16 + jitter(-2, 2, b), 13 + jitter(-1, 2, b)), -1),
            torch.stack((48 + jitter(-2, 2, b), 13 + jitter(-1, 2, b)), -1),
        ),
        1,
    )
    cells = torch.tensor([8.0, 24.0, 40.0, 56.0])
    object_xy = torch.stack(
        (cells + jitter(-1, 1, b, 4), 46 + jitter(-3, 3, b, 4)), -1
    )
    if attrs is None:
        attrs = torch.randint(4, (b, 4, 4), generator=g)
    else:
        attrs = torch.as_tensor(attrs).long().reshape(-1, 4, 4).expand(b, 4, 4).clone()
    return Scenes(kinds, machine_xy, attrs, object_xy)


def paired_view(generator, scenes, lamps, textures, cap=64):
    """Second view for S1 identity training (loss/generator knowledge only).

    New layouts and objects; the 2B machine body textures of the first view are
    permuted exactly onto the 2B machine places of the second view, and every
    machine's lamp is inverted relative to its source. Side and partner machine are
    therefore no matching shortcut. No second-view scene holds both machines of one
    first-view scene. Returns (scenes, lamps, textures, source) where `source` [B,2]
    is the flat first-view machine index (scene*2 + side) shown at each place.
    """
    b = len(scenes)
    if b < 2:
        raise ValueError("A paired view needs at least two scenes")
    for _ in range(cap):
        source = torch.randperm(2 * b, generator=generator).reshape(b, 2)
        if ((source[:, 0] // 2) != (source[:, 1] // 2)).all():
            break
    else:
        raise ValueError(f"Paired view found no valid permutation in {cap} draws")
    flat = source.flatten()
    shared = Textures(
        textures.colors.reshape(2 * b, 2, 3)[flat].reshape(b, 2, 2, 3),
        textures.pattern.flatten()[flat].reshape(b, 2),
        textures.period.flatten()[flat].reshape(b, 2),
    )
    view = sample_scenes(generator, scenes.kind.flatten()[flat].reshape(b, 2))
    return view, 1 - lamps.flatten()[flat].reshape(b, 2), shared, source


def _grid(device):
    y, x = torch.meshgrid(
        torch.arange(SIZE, device=device, dtype=torch.float32),
        torch.arange(SIZE, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return x, y


def render(scenes, lamps, textures=None):
    """Scenes [B], lamps [B,2] -> rgb [B,3,64,64] (8-bit exact), entity map [B,64,64].

    `textures` optionally overrides machine body textures (training nuisance only);
    the default renders each kind from the fixed texture table.
    """
    device = scenes.kind.device
    if textures is None:
        textures = kind_textures(scenes.kind.cpu())
    textures = Textures(*(v.to(device) for v in vars(textures).values()))
    lamps = torch.as_tensor(lamps, device=device).long().reshape(len(scenes), 2)
    x, y = _grid(device)
    b = len(scenes)
    rgb = torch.tensor(BACKGROUND, device=device)[None, :, None, None].repeat(
        b, 1, SIZE, SIZE
    )
    entity = torch.zeros(b, SIZE, SIZE, dtype=torch.long, device=device)
    for m in range(2):
        cx = scenes.machine_xy[:, m, 0, None, None]
        cy = scenes.machine_xy[:, m, 1, None, None]
        dx, dy = x - cx, y - cy
        body = (dx.abs() <= 11) & (dy.abs() <= 8)
        u, v = (dx + 11).long(), (dy + 8).long()
        p = textures.period[:, m][:, None, None]
        patterns = torch.stack(
            (
                ((u // p + v // p) % 2) == 0,
                (v // p) % 2 == 0,
                (u // p) % 2 == 0,
                ((u + v) // p) % 2 == 0,
                (u % p == 0) & (v % p == 0),
                (u >= 2) & (u <= 20) & (v >= 2) & (v <= 14),
            ),
            1,
        )
        choose = textures.pattern[:, m]
        first = patterns[torch.arange(b, device=device), choose]
        pair = textures.colors[:, m]  # [B,2,3]
        texture = torch.where(
            first[:, None], pair[:, 0, :, None, None], pair[:, 1, :, None, None]
        )
        panel = (dx.abs() <= 4) & (dy >= -8) & (dy <= -1)
        lamp = dx.square() + (dy + 4.5).square() <= 3.2**2
        on = lamps[:, m].bool()[:, None, None, None]
        lamp_color = torch.where(
            on,
            torch.tensor(LAMP[1], device=device)[None, :, None, None],
            torch.tensor(LAMP[0], device=device)[None, :, None, None],
        )
        value = torch.where(
            panel[:, None],
            torch.tensor(PANEL, device=device)[None, :, None, None],
            texture,
        )
        value = torch.where(lamp[:, None], lamp_color, value)
        rgb = torch.where(body[:, None], value, rgb)
        entity = torch.where(body, torch.full_like(entity, 1 + m), entity)
    for i in range(4):
        cx = scenes.object_xy[:, i, 0, None, None]
        cy = scenes.object_xy[:, i, 1, None, None]
        dx, dy = x - cx, y - cy
        color, shape, size, pattern = scenes.attrs[:, i].unbind(-1)
        r = (size + 3).float()[:, None, None]
        arm = torch.clamp(torch.div(r, 3, rounding_mode="floor"), min=1)
        shapes = torch.stack(
            (
                (dx.abs() <= r) & (dy.abs() <= r),
                dx.square() + dy.square() <= (r + 0.5).square(),
                (dy >= -r) & (dy <= r) & (dx.abs() <= (dy + r) / 2 + 0.5),
                ((dx.abs() <= r) & (dy.abs() <= arm))
                | ((dy.abs() <= r) & (dx.abs() <= arm)),
            ),
            1,
        )
        mask = shapes[torch.arange(b, device=device), shape]
        dark = torch.stack(
            (
                torch.zeros_like(x, dtype=torch.bool).expand(b, SIZE, SIZE),
                ((y.long() % 2) == 0).expand(b, SIZE, SIZE),
                ((x.long() % 2) == 0).expand(b, SIZE, SIZE),
                (((x.long() % 2) == 0) & ((y.long() % 2) == 0)).expand(b, SIZE, SIZE),
            ),
            1,
        )[torch.arange(b, device=device), pattern]
        value = PALETTE.to(device)[color][:, :, None, None] * torch.where(
            dark, 0.45, 1.0
        )[:, None]
        rgb = torch.where(mask[:, None], value, rgb)
        entity = torch.where(mask, torch.full_like(entity, 3 + i), entity)
    return (rgb * 255).round() / 255, entity


# ------------------------------------------------------------------ execution and utility


@dataclass(frozen=True)
class ActionRecord:
    """Exact executable press: target machine pixel, role-a pixel, role-b pixel."""

    machine_xy: tuple[float, float]
    a_xy: tuple[float, float]
    b_xy: tuple[float, float]

    def __post_init__(self):
        for xy in (self.machine_xy, self.a_xy, self.b_xy):
            if len(xy) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in xy
            ):
                raise ValueError("Action coordinates must be three finite (x, y) pairs")

    def as_dict(self):
        return dict(machine_xy=list(self.machine_xy), a_xy=list(self.a_xy), b_xy=list(self.b_xy))


@dataclass(frozen=True)
class Receipt:
    status: str  # ok, miss, same_object, budget_exceeded
    cost: float
    presses: int


@dataclass(frozen=True)
class TaskContract:
    success: float = 1.0
    false_stop: float = -1.0
    abstain: float = -0.25
    press_cost: float = 0.05
    budget: int = 2

    def utility(self, outcome, presses):
        """The single terminal utility used by planner and evaluator."""
        return {
            "success": self.success,
            "false_stop": self.false_stop,
            "abstain": self.abstain,
        }[outcome] - self.press_cost * presses

    def expected_stop(self, p_success, presses):
        return p_success * self.utility("success", presses) + (
            1 - p_success
        ) * self.utility("false_stop", presses)


class RuleWorld:
    """Environment and executor for one scene. Holds hidden rules; never give to agents."""

    def __init__(self, scenes, rules, lamps, contract):
        if len(scenes) != 1 or len(rules) != 2:
            raise ValueError("One scene with two machine rules")
        self.scenes, self.rules, self.contract = scenes, tuple(rules), contract
        self.lamps = torch.as_tensor(lamps).long().reshape(2).clone()
        self.presses = 0
        self.receipts = []
        _, self.entity = render(scenes, self.lamps[None])
        self.entity = self.entity[0]

    def frame(self):
        return render(self.scenes, self.lamps[None])[0][0]

    def resolve(self, xy):
        x, y = math.floor(xy[0]), math.floor(xy[1])
        if not (0 <= x < SIZE and 0 <= y < SIZE):
            return -1
        return int(self.entity[y, x])

    def press(self, record):
        coordinates = (*record.machine_xy, *record.a_xy, *record.b_xy)
        if len(coordinates) != 6 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in coordinates
        ):
            raise ValueError("Action coordinates must be three finite (x, y) pairs")
        if self.presses >= self.contract.budget:
            receipt = Receipt("budget_exceeded", 0.0, self.presses)
            self.receipts.append(receipt)
            return receipt
        self.presses += 1
        machine = self.resolve(record.machine_xy)
        a, b = self.resolve(record.a_xy), self.resolve(record.b_xy)
        status = "ok"
        if machine not in (1, 2) or a < 3 or b < 3:
            status = "miss"
        elif a == b:
            status = "same_object"
        else:
            m = machine - 1
            attrs = self.scenes.attrs[0]
            self.lamps[m] = outcome(
                self.rules[m], attrs[a - 3].tolist(), attrs[b - 3].tolist(), int(self.lamps[m])
            )
        receipt = Receipt(status, self.contract.press_cost, self.presses)
        self.receipts.append(receipt)
        return receipt

    def verify(self, goal):
        return tuple(self.lamps.tolist()) == tuple(goal)


class Actuator:
    """The only environment surface an agent may call: observe and press."""

    def __init__(self, world):
        self._world = world

    def frame(self):
        return self._world.frame()

    def press(self, record):
        return self._world.press(record)


# ------------------------------------------------------------------ goals and strata


class StratumUnavailable(RuntimeError):
    pass


PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]


def min_presses(rules, attrs, lamps, goal, budget):
    """Independent BFS over true rules; None when unreachable within budget."""
    frontier = {tuple(int(v) for v in lamps)}
    goal = tuple(int(v) for v in goal)
    seen = set(frontier)
    for depth in range(budget + 1):
        if goal in frontier:
            return depth
        following = set()
        for state in frontier:
            for m in (0, 1):
                for i, j in PAIRS:
                    new = list(state)
                    new[m] = outcome(rules[m], attrs[i], attrs[j], state[m])
                    following.add(tuple(new))
        frontier = following - seen
        seen |= following
    return None


def goal_for_stratum(rules, attrs, lamps, stratum, generator):
    want = {"already": 0, "reach1": 1, "reach2": 2, "unreachable": None}[stratum]
    options = [
        (a, b)
        for a in (0, 1)
        for b in (0, 1)
        if min_presses(rules, attrs, lamps, (a, b), 2) == want
    ]
    if not options:
        return None
    return options[int(torch.randint(len(options), (1,), generator=generator))]


@dataclass(frozen=True)
class Task:
    scene: Scenes
    lamps: torch.Tensor
    goal: tuple[int, int]
    rules: tuple[Rule, Rule]
    stratum: str
    attempts: int


def sample_task(generator, kinds, rules, stratum, cap=200, attrs=None):
    for attempt in range(1, cap + 1):
        scene = sample_scenes(generator, torch.as_tensor(kinds)[None], attrs=attrs)
        lamps = torch.randint(2, (2,), generator=generator)
        goal = goal_for_stratum(
            rules, scene.attrs[0].tolist(), lamps.tolist(), stratum, generator
        )
        if goal is not None:
            return Task(scene, lamps, goal, tuple(rules), stratum, attempt)
    raise StratumUnavailable(f"{stratum} unavailable after {cap} scenes")


def stratum_schedule(count):
    counts = {k: int(round(p * count)) for k, p in STRATA.items()}
    counts["reach1"] += count - sum(counts.values())
    return [k for k, n in counts.items() for _ in range(n)]


# ------------------------------------------------------------------ floors


def floors():
    """Balanced-accuracy floors of shortcut predictors on the full R1 input domain.

    Scored event: y for category/relation, Δ=y xor s for transitions (as in the
    spec's section 8, but with every executed press u=1). The ν floor is the best
    of copy-s, operator-only (knows family/operator and s, ignores matching),
    majority-label and constant-event predictors, averaged over the family's rules.
    """
    domain = torch.cartesian_prod(*[torch.arange(4)] * 8, torch.arange(2))
    a, b, s = domain[:, :4], domain[:, 4:8], domain[:, 8]
    rules = grammar()
    tensors = rule_tensors(rules)
    result = {f: {"copy_s": [], "operator_only": [], "majority_label": [], "constant": []} for f in FAMILIES}
    for index, rule in enumerate(rules):
        t = {k: v[index].expand(len(domain), *v.shape[1:]) for k, v in tensors.items()}
        y = outcome_batch(t, a, b, s)
        transition = rule.family in ("open", "close", "toggle")
        event = (y ^ s) if transition else y
        operator = {
            "category": torch.zeros_like(y),
            "relation": torch.zeros_like(y),
            "open": torch.ones_like(y),
            "close": torch.zeros_like(y),
            "toggle": 1 - s,
        }[rule.family]
        majority = torch.full_like(y, int(y.float().mean() >= 0.5))
        for name, prediction in (
            ("copy_s", s),
            ("operator_only", operator),
            ("majority_label", majority),
        ):
            predicted = (prediction ^ s) if transition else prediction
            result[rule.family][name].append(balanced_accuracy(predicted, event))
        result[rule.family]["constant"].append(0.5)
    summary = {}
    for family, values in result.items():
        means = {k: sum(v) / len(v) for k, v in values.items()}
        means["floor"] = max(means.values())
        summary[family] = means
    return summary


def balanced_accuracy(predicted, target):
    predicted, target = predicted.bool(), target.bool()
    if target.all() or (~target).all():
        return float("nan")
    tpr = (predicted & target).sum() / target.sum()
    tnr = (~predicted & ~target).sum() / (~target).sum()
    return float((tpr + tnr) / 2)


# ------------------------------------------------------------------ training episodes


@dataclass
class Transitions:
    scene: torch.Tensor
    machine: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    pre: torch.Tensor  # [n,2] both lamps before
    post: torch.Tensor  # [n,2]
    outcome: torch.Tensor  # target machine after
    episode: torch.Tensor
    step: torch.Tensor = field(default=None)

    def __post_init__(self):
        if self.step is None:
            self.step = torch.zeros_like(self.scene)

    def __len__(self):
        return len(self.scene)


def _transitions(rows):
    if not rows:
        empty = torch.zeros(0, dtype=torch.long)
        return Transitions(empty, empty, empty, empty, empty.reshape(0, 2), empty.reshape(0, 2), empty, empty, empty)
    columns = list(zip(*rows))
    return Transitions(
        *(torch.tensor(c) for c in columns[:4]),
        torch.tensor(columns[4]),
        torch.tensor(columns[5]),
        *(torch.tensor(c) for c in columns[6:]),
    )


@dataclass
class EpisodeBatch:
    scenes: Scenes
    episode_of_scene: torch.Tensor
    target_machine: torch.Tensor  # [S] which machine of each scene shows the target kind
    target_kind: torch.Tensor  # [E] train-only label (keys InfoNCE)
    rules: tuple
    family: torch.Tensor  # [E] loss/evaluation only
    n_support: torch.Tensor
    support: Transitions
    query: Transitions
    chain: Transitions


def _press(rule, attrs, i, j, pre, machine):
    post = list(pre)
    post[machine] = outcome(rule, attrs[i], attrs[j], pre[machine])
    return post


def sample_episodes(
    generator,
    rules,
    kinds,
    *,
    episodes,
    support=(8, 16, 32, 64, 128),
    queries=32,
    p_empty=0.05,
    per_scene=8,
    chain=True,
):
    """Random support/query/chain transitions for core training or validation.

    Each episode draws a fresh kind->rule assignment. Demonstrators set both lamps
    uniformly before each press; pairs are uniform ordered object pairs. Support and
    query inputs are symbolically disjoint in (a attributes, b attributes, s).
    """
    g = generator
    kinds = list(kinds)
    if len(kinds) < 2:
        raise ValueError("Need at least two kinds")
    order = torch.randperm(len(kinds), generator=g).tolist()
    scenes, of_scene, target_m = [], [], []
    rows = {"support": [], "query": [], "chain": []}
    targets, chosen_rules, counts = [], [], []

    def new_scene(e, target):
        others = [k for k in kinds if k != target]
        other = others[int(torch.randint(len(others), (1,), generator=g))]
        side = int(torch.randint(2, (1,), generator=g))
        pair = [other, other]
        pair[side] = target
        scenes.append(sample_scenes(g, torch.tensor([pair])))
        of_scene.append(e)
        target_m.append(side)
        return len(scenes) - 1, side

    def draw(scene_index, side, rule, pre=None):
        attrs = scenes[scene_index].attrs[0].tolist()
        i, j = PAIRS[int(torch.randint(len(PAIRS), (1,), generator=g))]
        pre = torch.randint(2, (2,), generator=g).tolist() if pre is None else pre
        post = _press(rule, attrs, i, j, pre, side)
        return (scene_index, side, i, j, pre, post, post[side]), attrs

    for e in range(episodes):
        target = kinds[order[e % len(order)]]
        rule = rules[int(torch.randint(len(rules), (1,), generator=g))]
        n = 0 if float(torch.rand(1, generator=g)) < p_empty else support[
            int(torch.randint(len(support), (1,), generator=g))
        ]
        targets.append(target)
        chosen_rules.append(rule)
        counts.append(n)
        seen = set()
        remaining = n
        while remaining > 0:
            index, side = new_scene(e, target)
            for _ in range(min(per_scene, remaining)):
                row, attrs = draw(index, side, rule)
                rows["support"].append((*row, e, 0))
                seen.add((tuple(attrs[row[2]]), tuple(attrs[row[3]]), row[4][side]))
            remaining -= per_scene
        query_scenes = [new_scene(e, target) for _ in range(max(2, math.ceil(queries / per_scene)))]
        for q in range(queries):
            index, side = query_scenes[q % len(query_scenes)]
            for _ in range(50):
                row, attrs = draw(index, side, rule)
                if (tuple(attrs[row[2]]), tuple(attrs[row[3]]), row[4][side]) not in seen:
                    break
            else:
                raise ValueError("Cannot draw a query disjoint from the support inputs")
            rows["query"].append((*row, e, 0))
        if chain:
            # Rollout targets obey the same support/query disjointness contract.
            index, side = query_scenes[0]
            steps, pre = [], None
            for _ in range(2):
                for _ in range(50):
                    row, attrs = draw(index, side, rule, pre=pre)
                    if (tuple(attrs[row[2]]), tuple(attrs[row[3]]), row[4][side]) not in seen:
                        break
                else:
                    raise ValueError("Cannot draw a rollout step disjoint from the support inputs")
                steps.append(row)
                pre = row[5]
            rows["chain"].append((*steps[0], e, 0))
            rows["chain"].append((*steps[1], e, 1))
    return EpisodeBatch(
        Scenes.cat(scenes),
        torch.tensor(of_scene),
        torch.tensor(target_m),
        torch.tensor(targets),
        tuple(chosen_rules),
        torch.tensor([FAMILY_INDEX[r.family] for r in chosen_rules]),
        torch.tensor(counts),
        _transitions(rows["support"]),
        _transitions(rows["query"]),
        _transitions(rows["chain"]),
    )


def manifest():
    """Complete data identity written by each run."""
    groups = split_groups()
    textures = dict(
        colors=KIND_COLORS.tolist(),
        pattern=KIND_PATTERN.tolist(),
        period=KIND_PERIOD.tolist(),
    )
    from pathwm.io import digest

    return dict(
        generator="rule_world_64_v1",
        families=FAMILIES,
        rules=[r.key() for r in grammar()],
        split_groups=groups,
        split_sha256=digest(groups),
        kind_split={k: list(v) for k, v in KIND_SPLIT.items()},
        textures_sha256=digest(textures),
        strata=STRATA,
        contract=vars(TaskContract()),
        semantics="u=1 for executed presses; no drift; non-targeted machine unchanged",
    )
