"""Editable CPU foundation exercise, not a real-image or general-agent benchmark.

Two supplied object descriptors, an observed binary property, then identical final
features with that property hidden. Supervise keys/update and read through the real
thinker. Durable inference separately exercises graph, correction and replay.
"""

import argparse
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.models.modalities import (
    ImageEncoder,
    TextEncoder,
    ImageDecoder,
    AudioDecoder,
    TextDecoder,
)
from pathwm.models.multiscale import MultiScaleAudioEncoder, MultiScaleImageEncoder
from pathwm.world_state.modules import (
    Candidate,
    CandidateEncoder,
    AssociationScorer,
    AssociationBinder,
    RecurrentUpdater,
    ContextEncoder,
    TransitionPredictor,
)
from pathwm.world_state.session import WorldSession
from pathwm.world_state.retrieval import Query, intervene
from pathwm.world_state.extensions import (
    create_prototype,
    ControlBinding,
    Feedback,
    ActionProposal,
    select_action,
)
from pathwm.world_state.inspection import WorldTrace, inspect_store
from pathwm.io import (
    Run,
    seed_everything,
    atomic_json,
    evaluation_mode,
    training_mode,
    resume_arguments,
)
from pathwm.evaluation.report import write_report


class FoundationModel(nn.Module):
    def __init__(self):
        super().__init__()
        width = 16
        self.encoder = CandidateEncoder(7, 4, 4)
        self.scorer = AssociationScorer(4)
        self.updater = RecurrentUpdater(4, 8)
        self.context = ContextEncoder(
            width,
            {"state": nn.Linear(8, width)},
            {"state": ("belief", "state-v1")},
            max_tokens=8,
        )
        self.predictor = TransitionPredictor(8, 1)
        self.state_head = nn.Linear(8, 2)
        self.readout = nn.Linear(width, 2)
        self.agent = BeliefAgent(
            width=width,
            context_tokens=4,
            latent_groups=4,
            latent_codes=4,
            evidence_tokens=4,
            encoders={"image": ImageEncoder(width), "text": TextEncoder(width)},
            decoders={"image": ImageDecoder(width, 16)},
            updater=BeliefCorrection(width, 4, 4, 2),
            dynamics=BeliefDynamics(width, 4, 4, 2),
            thinker=Thinker(width),
            memory=HybridMemory(width, recent=2, block=2, blocks=1, latent_codes=4),
            action_head=ActionHead(width),
            monitor=ErrorMonitor(width),
        )
        # Complete modality interface; specialist training remains explicit.
        self.agent.encoders["audio"] = MultiScaleAudioEncoder(32, width)
        self.agent.encoders["video"] = MultiScaleImageEncoder(width, video=True)
        self.agent.decoders["audio"] = AudioDecoder(width, 32)
        self.agent.decoders["text"] = TextDecoder(width)
        self.agent.requires_grad_(False)
        self.agent.thinker.requires_grad_(True)

    def session(self, snapshot=None):
        self.eval()
        modules = dict(
            agent=self.agent,
            binder=AssociationBinder(self.scorer),
            updater=self.updater,
            context_encoder=self.context,
        )
        return (
            WorldSession(**modules)
            if snapshot is None
            else WorldSession.restore(snapshot, **modules)
        )


def features(ids, bits, *, visible=True):
    identity = F.one_hot(ids, 2).float()
    state = F.one_hot(bits, 2).float() if visible else torch.zeros(len(ids), 2)
    return torch.cat(
        (
            identity,
            state,
            torch.full((len(ids), 1), float(visible)),
            torch.zeros(len(ids), 2),
        ),
        1,
    )


def objective(model, ids, bits, *, trace=None):
    source = features(ids, bits)
    hidden = features(ids, bits, visible=False)
    keys, values = model.encoder(source)
    alternate_keys, _ = model.encoder(features(ids, 1 - bits))
    hidden_keys, _ = model.encoder(hidden)
    negative_keys, _ = model.encoder(features(1 - ids, bits))
    matching = (model.scorer(keys, alternate_keys).diagonal() - 1).square().mean() + (
        model.scorer(keys, negative_keys).diagonal() + 1
    ).square().mean()
    matching = (
        matching + (model.scorer(keys, hidden_keys).diagonal() - 1).square().mean()
    )
    first = model.updater(torch.zeros(len(ids), 8), values, 0.0)
    _, last_values = model.encoder(hidden)
    final = model.updater(first, last_values, 1.0)
    state_loss = F.cross_entropy(model.state_head(final), bits)
    # Exactly the context projection used after persistent retrieval; commits detach.
    tokens = model.context.project_value("state", final, age=0.0)[:, None]
    agent_state = model.agent.initial_state(len(ids), session_id="training-query")
    working = model.agent.think(agent_state, goal=tokens, steps=1, trace=trace)
    logits = model.readout(working.tokens[:, model.agent.layout["working"]].mean(1))
    read_loss = F.cross_entropy(logits, bits)
    action = (1 - bits).float()[:, None]
    prediction, scale = model.predictor(final, action, 1.0)
    prediction_loss = (
        F.mse_loss(model.state_head(prediction), F.one_hot(1 - bits, 2).float())
        + 0.01 * scale.mean()
    )
    total = matching + state_loss + read_loss + prediction_loss
    metrics = dict(
        loss=float(total.detach()),
        matching=float(matching.detach()),
        state_loss=float(state_loss.detach()),
        read_loss=float(read_loss.detach()),
        prediction_loss=float(prediction_loss.detach()),
        accuracy=float((logits.argmax(-1) == bits).float().mean()),
    )
    return total, metrics


@torch.no_grad()
def exercise(model, path, trace):
    session = model.session()
    empty_snapshot = session.snapshot()

    def candidates(bits, visible):
        ids = torch.arange(2)
        x = features(ids, torch.tensor(bits), visible=visible)
        keys, values = model.encoder(x)
        return tuple(
            Candidate(
                str(i),
                "supplied-regions",
                "features",
                keys[i],
                values[i],
                "descriptor",
                "encoder-v1",
                exclusive_group="frame",
            )
            for i in range(2)
        )

    first = session.observe(
        "first",
        occurred_at=1,
        available_at=1,
        candidates=candidates([0, 1], True),
        trace=trace,
    )
    second = session.observe(
        "hidden",
        occurred_at=2,
        available_at=2,
        candidates=candidates([0, 1], False),
        trace=trace,
    )
    session.save(path / "session.pt")
    restored = model.session(torch.load(path / "session.pt", weights_only=True))
    replay_equal = (
        restored.store.snapshot() == session.store.snapshot()
        and torch.equal(restored.state.logits, session.state.logits)
    )
    owners = [b["entity_id"] for b in first["bindings"] if b["entity_id"] is not None]
    result = dict(
        restore_equal=replay_equal,
        bindings_first=first["bindings"],
        bindings_hidden=second["bindings"],
        owners=len(set(owners)),
    )
    if len(set(owners)) == 2:
        a, b = owners
        # Read the same query from both saved sessions: same selected components/tokens.
        working, context, encoded = session.think(Query(entity_ids=(a,)), trace=trace)
        replay, _, encoded_replay = restored.think(Query(entity_ids=(a,)))
        result["exact_reasoner_replay"] = torch.equal(
            working.tokens, replay.tokens
        ) and torch.equal(encoded.values, encoded_replay.values)
        logits = model.readout(working.tokens[:, model.agent.layout["working"]].mean(1))
        altered = intervene(
            context,
            replacements={
                c.id: torch.zeros(c.shape)
                for c in context.components
                if c.name == "state"
            },
        )
        # Same pre-read state for a meaningful local intervention.
        alternative = model.session(torch.load(path / "session.pt", weights_only=True))
        changed, _, _ = alternative.think(
            Query(entity_ids=(a,)), context=altered, trace=trace
        )
        changed_logits = model.readout(
            changed.tokens[:, model.agent.layout["working"]].mean(1)
        )
        result["zero_memory_logit_change"] = float(
            (logits - changed_logits).square().mean()
        )
        result["runtime_answer"] = int(logits.argmax(-1))
        paired = model.session(empty_snapshot)
        paired.observe(
            "first", occurred_at=1, available_at=1, candidates=candidates([1, 0], True)
        )
        paired.observe(
            "hidden",
            occurred_at=2,
            available_at=2,
            candidates=candidates([1, 0], False),
        )
        paired_state, _, _ = paired.think(Query(entity_ids=(a,)))
        paired_logits = model.readout(
            paired_state.tokens[:, model.agent.layout["working"]].mean(1)
        )
        result["paired_history_answer"] = int(paired_logits.argmax(-1))
        result["paired_history_answers_correct"] = (
            result["runtime_answer"] == 0 and result["paired_history_answer"] == 1
        )
        proof = session.store.evidence()[0].id
        tx = session.store.begin(
            "mistaken-link", occurred_at=3, available_at=3, kind="correction"
        )
        link = tx.merge(a, b, evidence=(proof,))
        session.commit(tx, trace=trace)
        assert session.store.canonical(a) == session.store.canonical(b)
        tx = session.store.begin(
            "repair-link", occurred_at=4, available_at=4, kind="correction"
        )
        tx.split(link)
        session.commit(tx, trace=trace)
        result["split_restores_distinct_ids"] = session.store.canonical(
            a
        ) != session.store.canonical(b)
        tx = session.store.begin(
            "prototype", occurred_at=5, available_at=5, kind="internal"
        )
        concept = create_prototype(
            tx,
            [session.store.latest(e, "recognition") for e in owners],
            label="example group",
        )
        self_id = tx.create_entity("agent", kind="self")
        tx.put_component(
            self_id,
            "resources",
            torch.tensor([1.0]),
            space="explicit",
            model_version="1",
            role="inferred",
            data={"meaning": "one available query"},
        )
        session.commit(tx, trace=trace)
        result["prototype"] = concept
        control = ControlBinding(
            self_id, sensors=("supplied-regions",), actions=("inspect",)
        )
        proposal = select_action(
            [ActionProposal("inspect", cost=0.1, novelty=0.5)], control, mode="novelty"
        )
        result["proposed_action_not_executed"] = proposal.action
        trace.record(
            "feedback",
            {
                "features": Feedback(prediction_error=0.2).features().tolist(),
                "note": "supplied example, not inferred emotion",
            },
        )
    session.store.save(path / "world_snapshot.json")
    session.save(path / "session_after_correction.pt")
    atomic_json(path / "world_state.json", inspect_store(session.store))
    trace.export(path)
    return result


def run(args):
    seed_everything(args.seed)
    model = FoundationModel()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    settings = dict(
        seed=args.seed,
        steps=args.steps,
        purpose="development",
        batch_size=16,
        learning_rate=0.01,
        objective="matching + state + actual thinker readout + supervised toy action forecast",
        scope="supplied descriptors; mechanics and development learning only",
    )
    output = args.resume or args.output
    runner = Run(
        output,
        settings=settings,
        data={
            "kind": "four identity/bit combinations, hidden final property",
            "sha256": "procedural-world-foundation-v1",
        },
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device="cpu",
        resume=args.resume is not None,
    )
    trace = WorldTrace(max_records=400, tensor_values=8192)
    try:
        while runner.step < args.steps:
            if args.stop_after is not None and runner.step >= args.stop_after:
                break
            ids = torch.tensor(runner.sample(2, 16))
            bits = torch.tensor(runner.sample(2, 16))
            training_mode(model)
            optimizer.zero_grad()
            if runner.step in (0, args.steps - 1):
                with trace.capture(
                    model,
                    [
                        "encoder.key",
                        "updater.cell",
                        "context.projections.state",
                        "readout",
                        "predictor.network",
                    ],
                    gradients=True,
                ):
                    loss, metrics = objective(model, ids, bits, trace=trace)
                    loss.backward()
                trace.parameters(model)
            else:
                loss, metrics = objective(model, ids, bits)
                loss.backward()
            if not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            runner.step += 1
            runner.log(dict(step=runner.step, split="train", **metrics))
            trace.record("training", dict(step=runner.step, **metrics))
            if runner.step in (1, args.steps):
                trace.export(runner.path / "debug" / f"step_{runner.step:04d}")
        runner.save()
        with evaluation_mode(model):
            _, metrics = objective(
                model, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
            )
            diagnostics = exercise(model, runner.path, trace)
        initial = next(row["loss"] for row in runner.rows if row["split"] == "train")
        last = [row["loss"] for row in runner.rows if row["split"] == "train"][-1]
        gate = (
            last <= 0.9 * initial
            and diagnostics["restore_equal"]
            and diagnostics.get("exact_reasoner_replay", False)
        )
        atomic_json(
            runner.path / "result.json",
            dict(
                evaluation_scope="CPU development: supplied candidates and four property/identity combinations; no independent capability test",
                metrics={"functional": metrics, "runtime": diagnostics},
                gate=gate,
                limitations=[
                    "No raw-image object discovery",
                    "No learned concept induction",
                    "No general dynamics/uncertainty/exploration validation",
                ],
            ),
        )
        complete = runner.step == args.steps
        runner.status("completed" if complete else "paused", "pending")
        write_report(runner.path)
        runner.status("completed" if complete else "paused", "complete")
        print(
            json.dumps(
                dict(
                    output=str(runner.path),
                    step=runner.step,
                    gate=gate,
                    loss_ratio=last / initial,
                    diagnostics=diagnostics,
                ),
                indent=2,
            )
        )
    except Exception as error:
        runner.status("failed", "incomplete", str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/world_state_foundation_v1/development"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=61201)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check", action="store_true")
    args = resume_arguments(parser, parser.parse_args())
    if args.check:
        args.stop_after = 4
    if args.steps < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("positive update counts required")
    run(args)


if __name__ == "__main__":
    main()
