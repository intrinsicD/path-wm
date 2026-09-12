"""One editable multimodal world-model recipe, initially a CPU development run.

Synthetic synchronized images/video, impact waveforms and short byte descriptions
exercise every adapter. --dataset pusht uses actual observations/actions and omits
audio/text. Defaults are software-development budgets, not capability benchmarks.
"""

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
from time import perf_counter
import wave

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.models.modalities import (
    Observation,
    ImageDecoder,
    AudioDecoder,
    TextDecoder,
    bytes_batch,
    bytes_text,
)
from pathwm.models.multiscale import (
    MultiScaleImageEncoder,
    MultiScaleAudioEncoder,
    MultiScaleTextEncoder,
    FeatureController,
)
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.models.agent_state import EpisodicMemory
from pathwm.models.tasks import (
    Actor,
    OutputControl,
    OutputRequest,
    TaskRequest,
    TaskSession,
    OPERATIONS,
    MetadataEncoder,
    TaskInterpreter,
    TaskPolicy,
)
from pathwm.models.agent import (
    MultimodalAgent,
    ObservationUpdate,
    LatentDynamics,
    Thinker,
    ActionHead,
    ErrorMonitor,
)
from pathwm.data.sequences import PushTSequences
from pathwm.evaluation.agent import plan
from pathwm.evaluation.report import write_report, render_report
from pathwm.io import (
    Run,
    atomic_json,
    digest,
    evaluation_mode,
    seed_everything,
    training_mode,
    trainable_parameters,
    resume_arguments,
    state_hash,
    file_hash,
    load_component,
)
from pathwm.training.improvement import replay_probabilities, try_improvement
from pathwm.models.recall import (
    RecallHead,
    RecallQuery,
    SeenRecord,
    historical_target,
    recall_logits,
    select_recall,
    verify_recall,
    LABELS,
)
from pathwm.evaluation.recall import (
    fit_temperature,
    recall_metrics,
    confidence_diagnostics,
)
from pathwm.data.entities import (
    EntityEpisodes,
    EntityMatches,
    VariableEntityMatches,
    NAMES as ENTITY_NAMES,
)
from pathwm.models.entities import EntityReader, SharedEntityReader, EntityMatchReader
from pathwm.evaluation.entities import entity_metrics, association_diagnostics
from pathwm.models.facts import FactReader, EventFactReader
from pathwm.evaluation.facts import fact_metrics, extraction_gates, binding_reference


def build_model(
    width=32,
    image_size=16,
    audio_samples=32,
    *,
    code_width=16,
    levels=3,
    cross_scale=True,
    state_model="gaussian",
    memory_recent=32,
    memory_block=8,
    memory_blocks=16,
    recall=False,
    facts=False,
    entities=False,
    entity_matching=False,
    entity_association="raw",
    entity_reader="recurrent",
    fact_reader="direct",
    fact_encoder_weights=None,
):
    if entity_matching:
        return EntityMatchReader(width)
    if entities:
        if entity_reader == "shared":
            if entity_association not in ("observed", "learned"):
                raise ValueError("Shared entity reader requires observed association")
            return SharedEntityReader(width, entity_association)
        return EntityReader(width, entity_association)
    if fact_encoder_weights is not None and (not facts or fact_reader != "event"):
        raise ValueError("Fact encoder weights require the event fact reader")
    if facts:
        direct = FactReader(width)
        if fact_reader == "direct":
            return direct
        if fact_reader != "event" or state_model != "belief":
            raise ValueError("Event fact reader requires the categorical belief model")
    elif fact_reader != "direct":
        raise ValueError("Fact reader choice requires the facts dataset")
    if state_model not in ("gaussian", "belief"):
        raise ValueError("Unknown state model")
    factory = BeliefAgent if state_model == "belief" else MultimodalAgent
    options = (
        dict(
            context_tokens=16,
            latent_groups=8,
            latent_codes=8,
            evidence_tokens=8,
            time_unit="steps",
        )
        if state_model == "belief"
        else {}
    )
    # Replace a constructor here to compare components; no registration is needed.
    model = factory(
        **options,
        width=width,
        encoders={
            "image": MultiScaleImageEncoder(
                width, code_width=code_width, levels=levels, cross_scale=cross_scale
            ),
            "video": MultiScaleImageEncoder(
                width,
                video=True,
                code_width=code_width,
                levels=levels,
                cross_scale=cross_scale,
            ),
            "audio": MultiScaleAudioEncoder(
                audio_samples,
                width,
                code_width=code_width,
                levels=levels,
                cross_scale=cross_scale,
            ),
            "text": MultiScaleTextEncoder(
                width, code_width=code_width, levels=levels, cross_scale=cross_scale
            ),
        },
        decoders={
            "image": ImageDecoder(width, image_size),
            "audio": AudioDecoder(width, audio_samples),
            "text": TextDecoder(width),
        },
        updater=BeliefCorrection(width, 8, 8, 2)
        if state_model == "belief"
        else ObservationUpdate(width),
        dynamics=BeliefDynamics(width, 8, 8, 2)
        if state_model == "belief"
        else LatentDynamics(width),
        thinker=Thinker(width),
        memory=HybridMemory(
            width, recent=memory_recent, block=memory_block, blocks=memory_blocks
        )
        if state_model == "belief"
        else EpisodicMemory(capacity=16, retrieve_count=2),
        action_head=ActionHead(width),
        monitor=ErrorMonitor(width),
        feature_controller=FeatureController(width, code_width),
        metadata_encoder=MetadataEncoder(width),
        task_interpreter=TaskInterpreter(width),
        task_policy=TaskPolicy(width),
    )
    if recall:
        if state_model != "belief":
            raise ValueError("Recall requires the categorical belief model")
        model.recall_head = RecallHead(width)
    if facts:
        model = EventFactReader(model, direct)
        if fact_encoder_weights is not None:
            path = Path(fact_encoder_weights).resolve()
            expected = file_hash(path)
            encoder = model.agent.encoders["text"]
            load_component(encoder, path, "agent.encoder")
            if file_hash(path) != expected:
                raise ValueError("Fact encoder checkpoint changed during loading")
            if any(not torch.isfinite(v).all() for v in encoder.state_dict().values()):
                raise ValueError("Nonfinite fact encoder checkpoint")
            model.encoder_initialization = dict(
                checkpoint=str(path),
                sha256=expected,
                component="agent.encoder",
                encoder_sha256=state_hash(encoder),
                trainable=True,
            )
        return model
    return model


def retention_tiers(length, recent, block, blocks):
    """Deterministic reference membership before looking at model predictions."""
    queue, staging, compressed, consolidated = [], [], [], []
    for ordinal in range(1, length + 1):
        queue.append(ordinal)
        if len(queue) > recent:
            staging.append(queue.pop(0))
        if len(staging) == block:
            compressed.append(staging)
            staging = []
            if len(compressed) > blocks:
                consolidated.extend(compressed.pop(0))
    return dict(
        recent=queue,
        staging=staging,
        compressed=[x for b in compressed for x in b],
        consolidated=consolidated,
    )


class FactExamples:
    """Finite semantic-pair split; labels never enter the model forward call."""

    def __init__(self, split, count):
        if split not in ("train", "validation"):
            raise ValueError("Fact control uses train/development only")
        self.split = split
        pairs = [
            (e, loc)
            for e in range(32)
            for loc in range(4)
            if ((e + loc) % 4 == 0) == (split == "validation")
        ]
        if count != len(pairs):
            raise ValueError(
                "Fact control requires exactly 96 train and 32 development pairs"
            )
        self.entities, self.locations = map(list, zip(*pairs))
        self.texts = [SeenRecord(1, e, loc).text for e, loc in pairs]
        self.manifest = [
            dict(entity=e, location=loc, text=t)
            for (e, loc), t in zip(pairs, self.texts)
        ]
        self.cohorts = self.groups = [split] * len(pairs)
        self.identity = dict(
            dataset="direct_facts_v1",
            split=split,
            count=count,
            rule="development iff (entity+location)%4==0",
            sha256=digest(self.manifest),
            episode_sha256=[digest(x) for x in pairs],
        )

    def __len__(self):
        return len(self.entities)

    def batch(self, indices, device="cpu"):
        if any(not 0 <= int(i) < len(self) for i in indices):
            raise ValueError("Fact example index out of bounds")
        ids, valid = bytes_batch([self.texts[int(i)] for i in indices], device)
        return dict(
            fact_observation=Observation(
                ids, torch.zeros_like(ids, dtype=torch.float64), valid
            ),
            entities=torch.tensor(
                [self.entities[int(i)] for i in indices], device=device
            ),
            locations=torch.tensor(
                [self.locations[int(i)] for i in indices], device=device
            ),
        )


@torch.no_grad()
def fact_predictions(model, data, settings, *, deadline=None):
    outputs = []
    with evaluation_mode(model):
        for start in range(0, len(data), settings["batch_size"]):
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Fact diagnostic active-time budget exhausted")
            batch = data.batch(
                range(start, min(start + settings["batch_size"], len(data))),
                settings["device"],
            )
            if settings.get("fact_reader", "direct") == "event":
                # One repeatable categorical realization, not a posterior expectation.
                # evaluation_mode restores the caller's RNG after the complete pass.
                torch.manual_seed(settings["seed"] + 1000000 + start)
            entity, location = model(batch["fact_observation"])
            outputs.append((entity.cpu(), location.cpu()))
    return dict(
        entity_logits=torch.cat([x[0] for x in outputs]),
        location_logits=torch.cat([x[1] for x in outputs]),
        entities=torch.tensor(data.entities),
        locations=torch.tensor(data.locations),
    )


@torch.no_grad()
def finish_facts(run, learner, training, validation, settings, deadline):
    identity = dict(
        model_sha256=state_hash(learner.agent),
        final_step=run.step,
        train=training.identity,
        development=validation.identity,
    )
    path = run.path / "fact_predictions.pt"
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if (
            cache.get("identity") != identity
            or cache.get("schema") != "pathwm-fact-predictions-v1"
        ):
            raise ValueError(
                "Fact prediction cache does not match final checkpoint/splits"
            )
    else:
        cache = dict(
            schema="pathwm-fact-predictions-v1",
            identity=identity,
            train=fact_predictions(
                learner.agent, training, settings, deadline=deadline
            ),
            development=fact_predictions(
                learner.agent, validation, settings, deadline=deadline
            ),
        )
        temporary = path.with_suffix(".partial")
        torch.save(cache, temporary)
        temporary.replace(path)
    views = {name: fact_metrics(**cache[name]) for name in ("train", "development")}
    gates = extraction_gates(views["train"], views["development"])
    binding = dict(status="skipped_failed_extraction_gate")
    if gates["extraction"]:
        combined = {
            k: torch.cat([cache[s][k] for s in ("train", "development")])
            for k in cache["train"]
        }
        binding, raw = binding_reference(**combined)
        temporary = run.path / "binding_predictions.partial"
        torch.save(raw, temporary)
        temporary.replace(run.path / "binding_predictions.pt")
    per_entity = {}
    for name in ("train", "development"):
        rows = cache[name]
        per_entity[name] = {
            str(e): fact_metrics(
                **{k: v[rows["entities"] == e] for k, v in rows.items()}
            )
            for e in range(32)
        }
    examples = [
        dict(
            **record,
            predicted_entity=int(cache["development"]["entity_logits"][i].argmax()),
            predicted_location=int(cache["development"]["location_logits"][i].argmax()),
        )
        for i, record in enumerate(validation.manifest)
    ]
    atomic_json(
        run.path / "fact_results.json",
        dict(
            schema="pathwm-fact-results-v1",
            **identity,
            reader=settings.get("fact_reader", "direct"),
            encoder_initialization=settings.get("fact_encoder_initialization"),
            views=views,
            gates=gates,
            per_entity=per_entity,
            binding=binding,
            examples=examples,
            scope=(
                "Fresh single-event agent state, ordinary task interpreter and two thinking/memory reads; heads read working tokens only. Fixed-seed categorical evaluation is one repeatable realization. No retention, learned entity-query binding, calibration or final-test claim."
                if settings.get("fact_reader", "direct") == "event"
                else "Same encoder architecture trained from scratch; direct extraction and an explicit binding reference. No recurrent model, calibration or final test."
            )
            + (
                " Shared text encoder initialized from a direct-fact checkpoint and kept trainable; all other parameters start fresh. Development combinations were inspected previously. Recipient curves exclude donor training exposure."
                if settings.get("fact_encoder_initialization")
                else ""
            ),
        ),
    )
    if not any(r["split"] == "diagnostic_development" for r in run.rows):
        for name, values in views.items():
            run.log(dict(step=run.step, split=f"diagnostic_{name}", **values))


@torch.no_grad()
def entity_predictions(model, data, settings, *, deadline=None):
    outputs = [[] for _ in data.targets]
    with evaluation_mode(model):
        for start in range(0, len(data), settings["batch_size"]):
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Entity diagnostic active-time budget exhausted")
            batch = data.batch(
                range(start, min(start + settings["batch_size"], len(data))),
                settings["device"],
            )
            for dest, score in zip(outputs, model(batch["entity_inputs"])):
                dest.append(score.cpu())
    return dict(logits=tuple(torch.cat(v) for v in outputs), targets=data.targets)


@torch.no_grad()
def finish_entities(run, learner, training, validation, settings, deadline):
    identity = dict(
        model_sha256=state_hash(learner.agent),
        final_step=run.step,
        train=training.identity,
        development=validation.identity,
    )
    path = run.path / "entity_predictions.pt"
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if (
            cache.get("identity") != identity
            or cache.get("schema") != "pathwm-entity-predictions-v1"
        ):
            raise ValueError("Entity prediction cache does not match checkpoint/splits")
    else:
        cache = dict(
            schema="pathwm-entity-predictions-v1",
            identity=identity,
            train=entity_predictions(
                learner.agent, training, settings, deadline=deadline
            ),
            development=entity_predictions(
                learner.agent, validation, settings, deadline=deadline
            ),
        )
        if settings.get("entity_association") == "learned":
            cache["association_diagnostics"] = {
                stage: {
                    name: association_diagnostics(model, data)
                    for name, data in (("train", training), ("development", validation))
                }
                for stage, model in (
                    ("initial", learner.target),
                    ("final", learner.agent),
                )
            }
        temporary = path.with_suffix(".partial")
        torch.save(cache, temporary)
        temporary.replace(path)
    scores = {
        name: entity_metrics(
            cache[name]["logits"], cache[name]["targets"], data.cohorts, data.groups
        )
        for name, data in (("train", training), ("development", validation))
    }
    examples = [
        dict(
            cohort=r["cohort"],
            group=r["group"],
            inputs=r["inputs"],
            targets=r["targets"],
            probabilities=[
                p[i].double().softmax(-1).tolist()
                for p in cache["development"]["logits"]
            ],
        )
        for i, r in enumerate(validation.manifest)
    ]
    association_note = ""
    if "association_diagnostics" in cache:
        initial = cache["association_diagnostics"]["initial"]["development"]
        final = cache["association_diagnostics"]["final"]["development"]
        association_note = (
            f" Association diagnostic on {final['descriptor_groups']} development descriptor groups: "
            f"action matching {initial['action_accuracy']:.1%} to {final['action_accuracy']:.1%}; "
            f"final matching {initial['final_accuracy']:.1%} to {final['final_accuracy']:.1%}. "
        )
    atomic_json(
        run.path / "entity_results.json",
        dict(
            schema="pathwm-entity-results-v1",
            **identity,
            scores=scores,
            final_view_bounds=validation.final_view_bounds(),
            examples=examples,
            association_diagnostics=cache.get("association_diagnostics"),
            association=settings.get("entity_association", "raw"),
            reader=settings.get("entity_reader", "recurrent"),
            task="matching" if settings["dataset"] == "entity-matching" else "tracking",
            descriptor_noise=settings.get("entity_noise", 0.0),
            scope=(
                "Known-versus-new matching; separated synthetic query distances, candidate count follows the recorded dataset; no allocation or open-world calibration. "
                if settings["dataset"] == "entity-matching"
                else association_note
                + f"Descriptor noise fraction: {settings.get('entity_noise', 0.0)}. Reader: {settings.get('entity_reader', 'recurrent')}. Association mode: {settings.get('entity_association', 'raw')}. Controlled candidate features; three observations; fixed candidate streams; no learned graph. Half the episodes hide final identity. No visual discovery, graph learning, motor control or independent final-test claim."
            ),
        ),
    )
    if not any(r["split"] == "diagnostic_development" for r in run.rows):
        run.log(
            dict(
                step=run.step,
                split="diagnostic_development",
                **scores["development"]["overall"],
            )
        )


class RecallEpisodes:
    """Complete independent text sessions; evaluator fields are separate from inputs."""

    def __init__(
        self,
        *,
        split,
        count,
        history,
        recent=32,
        block=8,
        blocks=16,
        truncate=32,
        abstain_cost=0.25,
        mode="history",
    ):
        split_seed = {
            "train": 12001,
            "validation": 22001,
            "calibration": 32001,
            "test": 42001,
        }
        if mode not in ("history", "current-recent"):
            raise ValueError("Unknown recall mode")
        diagnostic = mode == "current-recent"
        if diagnostic and split not in ("train", "validation"):
            raise ValueError("Recall diagnostic uses train/development only")
        multiple = 10 if diagnostic else 15
        if split not in split_seed or count < 1 or count % multiple or truncate < 0:
            raise ValueError(
                f"Recall splits require a positive multiple of {multiple} episodes and nonnegative truncation"
            )
        tiers = retention_tiers(history, recent, block, blocks)
        if diagnostic:
            if (
                split not in ("train", "validation")
                or history < 3
                or recent < 2
                or (0 < truncate < history)
            ):
                raise ValueError(
                    "Recall diagnostic requires train/development, history >= 3, recent >= 2 and full gradients"
                )
            tiers = dict(current=[history], recent=[history - 1])
        elif any(not tiers[k] for k in ("recent", "compressed", "consolidated")):
            raise ValueError(
                "Recall history must exercise recent, compressed and consolidated records"
            )
        self.items, self.records, self.groups, self.hidden_locations = [], [], [], []
        self.cohorts, self.mode = [], mode
        self.truncate, self.split = truncate, split
        seed = split_seed[split] + (1000000 if diagnostic else 0)
        rng = np.random.default_rng(seed)
        design = rng.permutation(count)
        cohorts = (
            ("current", "recent")
            if diagnostic
            else ("recent", "compressed", "consolidated")
        )
        for index, code in enumerate(design):
            label, cohort = (
                int(code % 5),
                cohorts[int(code // 5) % len(cohorts)],
            )
            entity = int(rng.integers(32))
            last = int(rng.choice(tiers[cohort])) if label < 4 else None
            others = [e for e in range(32) if e != entity]
            records = [
                SeenRecord(t + 1, int(rng.choice(others)), int(rng.integers(4)))
                for t in range(history)
            ]
            if last is not None:
                records[last - 1] = SeenRecord(last, entity, label)
                if last > 1:
                    earlier = int(rng.integers(1, last))
                    records[earlier - 1] = SeenRecord(
                        earlier,
                        entity,
                        (label + int(rng.integers(1, 4))) % 4
                        if diagnostic
                        else int(rng.integers(4)),
                    )
            session = (
                f"{'recall-diagnostic' if diagnostic else 'recall'}/{split}/{index}"
            )
            task = TaskRequest(
                f"{session}/query",
                f"Where was entity=e{entity:02d} last observed?",
                Actor("user", "fixture"),
            )
            query = RecallQuery(
                task, session, entity, history, abstain_cost=abstain_cost
            )
            self.items.append(
                dict(
                    records=tuple(records),
                    query=query,
                    time_offset=int(rng.integers(1000)),
                )
            )
            self.records.append(tuple(records))
            self.groups.append(cohort if label < 4 else "not_observed")
            self.cohorts.append(cohort)
            # Independent hidden trajectory is evaluator-only and cannot change visible records.
            self.hidden_locations.append(
                np.random.default_rng(split_seed[split] + 100000 + index)
                .integers(4, size=history)
                .tolist()
            )
            assert historical_target(records, query) == label
        manifest = [
            dict(
                session=x["query"].session_id,
                entity=x["query"].entity,
                records=[asdict(r) for r in x["records"]],
                offset=x["time_offset"],
            )
            for x in self.items
        ]
        self.manifest = manifest
        self.identity = dict(
            dataset="historical_recall_v1",
            split=split,
            count=count,
            history=history,
            reference_memory=dict(recent=recent, block=block, blocks=blocks),
            sha256=digest(manifest),
            labels=list(LABELS),
            class_counts=[count // 5] * 5,
            groups={g: self.groups.count(g) for g in set(self.groups)},
            input="canonical text; complete session; unmarked",
            seed=seed,
        )
        if diagnostic:
            # Exclude offsets/IDs: different metadata must not hide identical tasks.
            signatures = [
                digest(
                    dict(
                        entity=x["query"].entity,
                        records=[asdict(r) for r in x["records"]],
                    )
                )
                for x in self.items
            ]
            if len(set(signatures)) != len(signatures):
                raise ValueError(
                    "Duplicate diagnostic episodes; choose a new explicit population"
                )
            self.identity.update(
                dataset="current_recent_recall_v1",
                mode=mode,
                cohorts={c: self.cohorts.count(c) for c in cohorts},
                episode_sha256=signatures,
            )

    def __len__(self):
        return len(self.items)

    def batch(self, indices, device="cpu"):
        if any(not 0 <= int(i) < len(self) for i in indices):
            raise ValueError("Recall episode index out of bounds")
        inputs = [self.items[int(i)] for i in indices]
        return dict(
            recall_inputs=inputs,
            truncate=self.truncate,
            labels=torch.tensor(
                [historical_target(x["records"], x["query"]) for x in inputs],
                device=device,
            ),
            groups=[self.groups[int(i)] for i in indices],
        )


def recall_forward(model, inputs, *, truncate=32, auxiliary=False):
    """No label arguments. Whole prefix executes; gradients cover the last segment."""
    from pathwm.models.belief_state import Packet
    from pathwm.training.belief import split_kl

    records, query = inputs["records"], inputs["query"]
    if not records or records[-1].ordinal != query.cutoff:
        raise ValueError("Complete recall input must end at its query cutoff")
    if [r.ordinal for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Recall input must be a complete ordered session")
    device = next(model.parameters()).device
    state = model.initial_state(
        1, time=inputs["time_offset"], session_id=query.session_id
    )
    start = max(0, len(records) - truncate) if truncate else 0
    losses = []
    for t, record in enumerate(records):
        track = torch.is_grad_enabled() and t >= start
        with torch.set_grad_enabled(track):
            ids, valid = bytes_batch([record.text], device)
            timestamp = inputs["time_offset"] + record.ordinal
            observation = Observation(
                ids,
                torch.full(
                    ids.shape, float(timestamp), device=device, dtype=torch.float64
                ),
                valid,
            )
            pending = model.begin_event(
                state,
                event_id=f"record-{record.ordinal}",
                ordinal=record.ordinal,
                time=timestamp,
                replay=track,
            )
            pending = model.add_packet(
                pending, Packet(f"text-{record.ordinal}", "text", observation)
            )
            state = model.commit_event(pending)
            if auxiliary and t >= start:
                posterior = model.decoders["text"](state.tokens, ids[:, :-1])
                source = model.decoders["text"](state.evidence, ids[:, :-1])
                grounding = F.cross_entropy(
                    posterior.transpose(1, 2), ids[:, 1:], ignore_index=0
                )
                evidence = F.cross_entropy(
                    source.transpose(1, 2), ids[:, 1:], ignore_index=0
                )
                dyn, rep = split_kl(state.logits, state.prior_logits)
                losses.append(torch.stack((grounding, evidence, dyn, rep)))
    logits, working = recall_logits(model, state, query)
    return logits, working, torch.stack(losses).mean(0) if losses else None


def recall_objective(learner, batch):
    outputs, aux = [], []
    for inputs in batch["recall_inputs"]:
        logits, _, loss = recall_forward(
            learner.agent, inputs, truncate=batch["truncate"], auxiliary=True
        )
        outputs.append(logits)
        aux.append(loss)
    logits = torch.cat(outputs)
    errors = F.cross_entropy(logits, batch["labels"], reduction="none")
    text, source, dyn, rep = torch.stack(aux).mean(0)
    losses = dict(
        recall_nll=errors.mean(),
        text_grounding=0.05 * text,
        source_grounding=0.05 * source,
        dynamics_kl=0.01 * dyn,
        representation_kl=0.001 * rep,
    )
    return losses, errors.detach(), dict(recall_nll=errors.mean().detach())


@torch.no_grad()
def recall_predictions(model, data, settings, *, deadline=None):
    outputs, labels, examples = [], [], []
    with evaluation_mode(model):
        for index, inputs in enumerate(data.items):
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Recall diagnostic active-time budget exhausted")
            # Repeatable per-episode latent samples; independent of batching and pauses.
            torch.manual_seed(settings["seed"] + data.identity["seed"] + index)
            logits, state, _ = recall_forward(model, inputs, truncate=data.truncate)
            outputs.append(logits.cpu())
            labels.append(historical_target(data.records[index], inputs["query"]))
            examples.append(
                dict(
                    query=inputs["query"].to_dict(),
                    group=data.groups[index],
                    records=[asdict(r) for r in data.records[index]],
                    memory_tensor_bytes=model.memory.storage_bytes(state.memory),
                )
            )
    return torch.cat(outputs), torch.tensor(labels), examples


def diagnostic_metrics(logits, labels, examples, settings):
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite diagnostic factual logits")
    masks = {
        g: torch.tensor([x["group"] == g for x in examples])
        for g in ("current", "recent", "not_observed")
    }
    masks["seen"] = labels < 4
    cost = settings.get("abstain_cost", 0.25)
    return dict(
        overall=recall_metrics(logits, labels, abstain_cost=cost),
        groups={
            g: recall_metrics(logits[m], labels[m], abstain_cost=cost)
            for g, m in masks.items()
            if m.any()
        },
        classes={
            name: recall_metrics(
                logits[labels == i], labels[labels == i], abstain_cost=cost
            )
            for i, name in enumerate(LABELS)
            if (labels == i).any()
        },
    )


@torch.no_grad()
def finish_recall_diagnostic(run, learner, training, validation, settings, deadline):
    """One final checkpoint, raw probabilities, train/development only; cache for reports."""
    identity = dict(
        model_sha256=state_hash(learner.agent),
        final_step=run.step,
        train=training.identity,
        validation=validation.identity,
    )
    cache_path = run.path / "recall_diagnostic_predictions.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            cache.get("schema") != "pathwm-recall-diagnostic-predictions-v1"
            or cache["identity"] != identity
        ):
            raise ValueError(
                "Diagnostic result cache does not match final weights/splits"
            )
    else:
        cache = dict(
            schema="pathwm-recall-diagnostic-predictions-v1", identity=identity
        )
        for name, data in (("train", training), ("development", validation)):
            logits, labels, examples = recall_predictions(
                learner.agent, data, settings, deadline=deadline
            )
            cache[name] = dict(logits=logits, labels=labels, examples=examples)
        temporary = cache_path.with_suffix(".partial")
        torch.save(cache, temporary)
        temporary.replace(cache_path)
    views = {
        name: diagnostic_metrics(**cache[name], settings=settings)
        for name in ("train", "development")
    }
    train, dev = (views[name]["groups"] for name in ("train", "development"))
    fit = (
        all(train[g]["factual_accuracy"] >= 0.95 for g in ("current", "recent"))
        and train["seen"]["nll"] <= 0.35
    )
    fresh = all(
        dev[g]["factual_accuracy"] >= 0.8 for g in ("current", "recent")
    ) and dev["seen"]["nll"] < float(np.log(4))
    examples = []
    for i, source in enumerate(cache["development"]["examples"]):
        query = RecallQuery.from_dict(source["query"])
        decision = select_recall(
            cache["development"]["logits"][i].double().softmax(-1), query
        )
        examples.append(
            dict(
                **source,
                factual_prediction=LABELS[
                    int(cache["development"]["logits"][i].argmax())
                ],
                decision=decision.to_dict(),
                verification=verify_recall(
                    query,
                    decision,
                    tuple(SeenRecord(**r) for r in source["records"]),
                    query.session_id,
                ),
            )
        )
    atomic_json(
        run.path / "recall_diagnostic.json",
        dict(
            schema="pathwm-recall-diagnostic-v1",
            **identity,
            views=views,
            examples=examples,
            gates=dict(
                tiny_set_fit=fit,
                fresh_examples=fresh,
                advance_to_memory_comparison=fit and fresh,
                thresholds=dict(
                    train_seen_accuracy_each=0.95,
                    train_seen_nll_max=0.35,
                    development_seen_accuracy_each=0.8,
                    development_seen_nll_below=float(np.log(4)),
                ),
            ),
            calibration="not fitted; raw probabilities; final test not accessed",
            limits=[
                "One finite-budget pilot; not a capability or calibration guarantee.",
                "Fit failure does not distinguish insufficient budget from encoding/binding/optimization defects.",
                "Recent retention does not establish compression or hierarchy superiority.",
            ],
        ),
    )
    # Rebuilding a completed report must not add ledger rows or evaluate again.
    if not any(r["split"] == "diagnostic_development" for r in run.rows):
        for name, view in views.items():
            run.log(
                dict(
                    step=run.step,
                    split=f"diagnostic_{name}",
                    **view["overall"],
                    **{
                        f"group_{g}_{k}": row[k]
                        for g, row in view["groups"].items()
                        for k in ("factual_accuracy", "nll", "examples")
                    },
                )
            )


@torch.no_grad()
def finish_recall(run, learner, calibration, test, settings):
    """Fit on calibration only; cache test outputs once for all report views."""
    identity = dict(
        selected_model_sha256=state_hash(learner.target),
        selected_step=int(learner.best_step),
        calibration=calibration.identity,
        test=test.identity,
    )
    cache_path = run.path / "recall_predictions.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            cache.get("schema") != "pathwm-recall-predictions-v1"
            or cache["identity"] != identity
        ):
            raise ValueError("Recall result cache does not match selected model/splits")
    else:
        started = perf_counter()
        cal_logits, cal_labels, _ = recall_predictions(
            learner.target, calibration, settings
        )
        calibration_seconds = perf_counter() - started
        fit = fit_temperature(cal_logits, cal_labels)
        started = perf_counter()
        test_logits, test_labels, examples = recall_predictions(
            learner.target, test, settings
        )
        test_seconds = perf_counter() - started
        cache = dict(
            schema="pathwm-recall-predictions-v1",
            identity=identity,
            calibration_logits=cal_logits,
            calibration_labels=cal_labels,
            fit=fit,
            logits=test_logits,
            labels=test_labels,
            examples=examples,
            timing=dict(
                calibration_inference_seconds=calibration_seconds,
                test_inference_seconds=test_seconds,
                test_seconds_per_episode=test_seconds / len(test),
                events_per_episode=test.identity["history"],
                thinking_rounds_per_episode=2,
                factual_head_calls_per_episode=1,
                device=settings["device"],
                note="Wall-clock serial inference including Python and transfers; descriptive, not a benchmark",
            ),
        )
        temporary = cache_path.with_suffix(".partial")
        torch.save(cache, temporary)
        temporary.replace(cache_path)
    fit, logits, labels = cache["fit"], cache["logits"], cache["labels"]
    learner.temperature.fill_(fit["temperature"])
    cost = settings.get("abstain_cost", 0.25)
    masks = {
        g: torch.tensor([x["group"] == g for x in cache["examples"]])
        for g in ("recent", "compressed", "consolidated", "not_observed")
    }
    masks["seen"] = labels < 4
    masks["seen_old"] = masks["compressed"] | masks["consolidated"]
    views = {}
    for name, temperature in (("raw", 1.0), ("calibrated", fit["temperature"])):
        views[name] = dict(
            overall=recall_metrics(
                logits, labels, temperature=temperature, abstain_cost=cost
            ),
            groups={
                g: recall_metrics(
                    logits[m], labels[m], temperature=temperature, abstain_cost=cost
                )
                for g, m in masks.items()
                if m.any()
            },
            costs={
                str(c): recall_metrics(
                    logits, labels, temperature=temperature, abstain_cost=c
                )
                for c in (0.1, 0.25, 0.5)
            },
            diagnostics=confidence_diagnostics(logits, labels, temperature),
        )
    examples = []
    for index, source in enumerate(cache["examples"]):
        query = RecallQuery.from_dict(source["query"])
        decision = select_recall(
            (logits[index].double() / fit["temperature"]).softmax(-1), query
        )
        verification = verify_recall(
            query,
            decision,
            tuple(SeenRecord(**r) for r in source["records"]),
            query.session_id,
        )
        examples.append(
            dict(**source, decision=decision.to_dict(), verification=verification)
        )
    result = dict(
        schema="pathwm-recall-results-v1",
        **identity,
        development_nll=float(learner.best_nll),
        calibration_fit=fit,
        labels=list(LABELS),
        views=views,
        examples=examples,
        baselines=dict(
            all_abstain=cost,
            uniform_always_answer_expected=0.8,
            always_not_observed=float((labels != 4).double().mean()),
            oracle_absence_only=cost * float((labels < 4).double().mean()),
            oracle_history=0.0,
        ),
        limits=[
            "Small development population; no capability or calibration guarantee.",
            "Current recurrent state can retain history; this does not isolate hierarchy benefit.",
            "Only the final replay segment receives gradients; prefix memories stay in the forward pass.",
        ],
    )
    atomic_json(run.path / "recall_results.json", result)
    atomic_json(run.path / "recall_performance.json", cache["timing"])


class SyntheticEpisodes:
    """Deterministic, artificial data with split-specific seeds; no external assets.

    A colored soft ball has inertia, controlled acceleration and reflecting walls.
    Impact sound and left/right/hit descriptions share its timestamps. Each index
    is a separate generated episode, so train/validation never share trajectories.
    """

    def __init__(
        self,
        split="train",
        count=32,
        image_size=16,
        audio_samples=32,
        history=2,
        horizon=2,
    ):
        if (
            split not in ("train", "validation", "test")
            or min(count, history, horizon) < 1
        ):
            raise ValueError("Invalid synthetic split or population")
        seed = dict(train=101, validation=202, test=303)[split]
        generator = np.random.default_rng(seed)
        n, length = count, history + horizon
        pos = generator.uniform(0.12, 0.88, (n, 2)).astype("float32")
        velocity = generator.uniform(-0.12, 0.12, (n, 2)).astype("float32")
        color = generator.uniform(0.35, 1, (n, 3)).astype("float32")
        actions = generator.uniform(-1, 1, (n, length - 1, 2)).astype("float32")
        y, x = np.meshgrid(
            np.linspace(0, 1, image_size), np.linspace(0, 1, image_size), indexing="ij"
        )
        images, audio, descriptions = [], [], []
        hit = np.zeros(n, dtype=bool)
        for t in range(length):
            distance = (x[None] - pos[:, 0, None, None]) ** 2 + (
                y[None] - pos[:, 1, None, None]
            ) ** 2
            ball = np.exp(-distance / 0.008).astype("float32")
            images.append(ball[:, None] * color[:, :, None, None])
            phase = np.arange(audio_samples, dtype="float32") / 8000
            sound = 0.4 * np.sin(2 * np.pi * (400 + pos[:, :1] * 400) * phase)
            audio.append((sound * hit[:, None]).astype("float32"))
            descriptions.append(
                [
                    ("left" if p[0] < 0.5 else "right") + (" hit" if h else " move")
                    for p, h in zip(pos, hit)
                ]
            )
            if t < length - 1:
                velocity = 0.9 * velocity + 0.06 * actions[:, t]
                pos = pos + velocity
                contact = (pos < 0.08) | (pos > 0.92)
                hit = contact.any(1)
                velocity = np.where(contact, -velocity, velocity)
                pos = np.where(
                    pos < 0.08, 0.16 - pos, np.where(pos > 0.92, 1.84 - pos, pos)
                )
        text, _ = bytes_batch(
            [descriptions[t][i] for i in range(n) for t in range(length)]
        )
        self.arrays = dict(
            images=torch.from_numpy(np.stack(images, 1)),
            actions=torch.from_numpy(actions),
            audio=torch.from_numpy(np.stack(audio, 1)),
            text=text.reshape(n, length, -1),
        )
        self.history, self.horizon = history, horizon
        self.identity = dict(
            kind="synthetic-controlled-ball-v1",
            split=split,
            seed=seed,
            count=count,
            image_size=image_size,
            audio_samples=audio_samples,
            sample_rate=8000,
            history=history,
            horizon=horizon,
            time_unit="one generated transition",
            content_sha256=digest(
                {k: digest(v.tolist()) for k, v in self.arrays.items()}
            ),
        )

    def __len__(self):
        return len(self.arrays["images"])

    def batch(self, indices, device="cpu"):
        ids = torch.as_tensor(indices, dtype=torch.long)
        if ids.ndim != 1 or len(ids) < 1 or (ids < 0).any() or (ids >= len(self)).any():
            raise ValueError("Batch indices must be nonempty and in bounds")
        return {
            **{k: v[ids].to(device) for k, v in self.arrays.items()},
            "sources": [
                f"synthetic/{self.identity['split']}/episode-{int(i)}" for i in ids
            ],
        }


def instruction_curriculum(split, count):
    """Small, disclosed language-label exercise, not general instruction data.

    Templates are disjoint across splits. Label order is independent of scene
    generation; task IDs are constant, so IDs cannot reveal sample/operation labels.
    Explicit controls vary independently of the seven-operation cycle. Inference
    never sees labels, template IDs or a keyword parser.
    """
    templates = {
        "train": (
            (
                "Think through the scene first.",
                "Reason about this scene before responding.",
            ),
            (
                "Recall an earlier observation.",
                "Retrieve a previous scene from memory.",
            ),
            ("Imagine a possible future.", "Predict what happens next."),
            ("Propose a movement.", "Choose an action for the next step."),
            ("Respond with {outputs}.", "Create {outputs} for the answer."),
            ("Do that thing.", "I want something, but I cannot say what."),
            ("The task is complete; stop.", "Everything is done; finish."),
        ),
        "validation": (
            ("Think about the scene carefully.", "First reason about what is visible."),
            (
                "Recall what you observed before.",
                "Retrieve something from your memory.",
            ),
            ("Imagine what could happen later.", "Predict the following scene."),
            ("Propose the next action.", "Choose a movement now."),
            (
                "Give me {outputs} as your response.",
                "Your answer should contain {outputs}.",
            ),
            ("Make it how I want it.", "You know, the thing I meant."),
            ("This task is already complete.", "We are done; stop working."),
        ),
        "test": (
            ("Reason first about the scene.",),
            ("Recall the prior scene.",),
            ("Predict a later scene.",),
            ("Propose an action now.",),
            ("Please return {outputs}.",),
            ("Do whatever I was thinking.",),
            ("Stop: this is finished.",),
        ),
    }
    modalities = ("image", "audio", "text", "video")
    user = Actor("user", "synthetic-user")
    sessions, operation_targets, modality_targets = [], [], []
    for i in range(count):
        operation, cycle = i % len(OPERATIONS), i // len(OPERATIONS)
        mask = 1 + cycle % 15
        chosen = tuple(m for j, m in enumerate(modalities) if mask & (1 << j))
        variants = templates[split][operation]
        instruction = variants[cycle % len(variants)].format(
            outputs=" and ".join(chosen)
        )
        control_cycle = (cycle // 3) % 3
        controls = (
            ()
            if control_cycle == 0
            else (
                OutputControl(
                    modalities[cycle % 4],
                    "disabled" if control_cycle == 1 else "required",
                    user,
                ),
            )
        )
        sessions.append(
            TaskSession(TaskRequest("example-task", instruction, user, controls))
        )
        operation_targets.append(operation)
        modality_targets.append(
            [
                float(operation == OPERATIONS.index("emit") and m in chosen)
                for m in modalities
            ]
        )
    return (
        tuple(sessions),
        torch.tensor(operation_targets),
        torch.tensor(modality_targets),
    )


class InstructionEpisodes(SyntheticEpisodes):
    """The existing world-model recipe with additional synthetic task supervision."""

    def __init__(self, split="train", count=112, **kwargs):
        super().__init__(split=split, count=count, **kwargs)
        self.tasks, self.operation_targets, self.modality_targets = (
            instruction_curriculum(split, count)
        )
        self.identity = dict(
            self.identity,
            kind="synthetic-ball-and-instructions-v1",
            tasks_sha256=digest(
                {
                    "requests": [s.request.to_dict() for s in self.tasks],
                    "operations": self.operation_targets.tolist(),
                    "modalities": self.modality_targets.tolist(),
                }
            ),
            curriculum="disjoint template families; labels are loss-only; artificial operation descriptions",
        )

    def batch(self, indices, device="cpu"):
        result = super().batch(indices, device)
        ids = torch.as_tensor(indices, dtype=torch.long)
        return dict(
            result,
            tasks=[self.tasks[int(i)] for i in ids],
            task_operation=self.operation_targets[ids].to(device),
            task_modalities=self.modality_targets[ids].to(device),
        )


def task_losses(model, state, batch):
    prediction = model.task_predictions(state, batch["tasks"])
    targets, modalities = batch["task_operation"], batch["task_modalities"]
    complete = (targets == OPERATIONS.index("finish")).float()
    losses = {
        "task_operation_ce": F.cross_entropy(prediction.operation_logits, targets),
        "task_modality_bce": F.binary_cross_entropy_with_logits(
            prediction.modality_logits, modalities
        ),
        "task_completion_bce": F.binary_cross_entropy_with_logits(
            prediction.completion_logits, complete
        ),
    }
    diagnostics = {
        "task_operation_error": (prediction.operation_logits.argmax(-1) != targets)
        .float()
        .mean(),
        "task_modality_error": ((prediction.modality_logits >= 0) != modalities.bool())
        .float()
        .mean(),
        "task_completion_error": (
            (prediction.completion_logits >= 0) != complete.bool()
        )
        .float()
        .mean(),
    }
    return losses, diagnostics


@torch.no_grad()
def task_evaluation(model, data, settings):
    """Raw predictions, hard-gate audit, text mismatch control and lexical baseline."""
    references, ref_operations, ref_modalities = instruction_curriculum(
        "train", settings["train_windows"]
    )

    def words(s):
        return set(re.findall(r"\w+", s.lower()))

    ref_words = [words(s.request.instruction) for s in references]
    rows = []
    for start in range(0, len(data), settings["batch_size"]):
        ids = list(range(start, min(start + settings["batch_size"], len(data))))
        batch = data.batch(ids, settings["device"])
        state = observe_history(model, batch, settings["history"])
        prediction = model.task_predictions(state, batch["tasks"])
        mismatched = [
            TaskSession(
                replace(
                    s.request,
                    instruction=data.tasks[(i + 1) % len(data)].request.instruction,
                )
            )
            for i, s in zip(ids, batch["tasks"])
        ]
        shuffled = model.task_predictions(state, mismatched)
        for j, i in enumerate(ids):
            session = batch["tasks"][j]
            selected = prediction.select(session, Actor("agent", "model"), j)
            target_op = int(data.operation_targets[i])
            target_mask = data.modality_targets[i].bool().tolist()
            text_words = words(session.request.instruction)
            neighbor = max(
                range(len(references)),
                key=lambda k: (
                    len(text_words & ref_words[k])
                    / max(1, len(text_words | ref_words[k]))
                ),
            )
            raw_mask = (prediction.modality_logits[j] >= 0).tolist()
            active = [r.modality for r in selected.requests]
            violations = sum(
                c.modality in active
                for c in session.request.controls
                if c.mode == "disabled"
            )
            violations += sum(
                c.modality not in active
                for c in session.request.controls
                if c.mode == "required" and selected.operation == "emit"
            )
            violations += int(
                selected.operation == "finish" and bool(session.remaining)
            )
            raw_violations = int(
                selected.raw_operation == "finish" and bool(session.remaining)
            )
            if selected.raw_operation == "emit":
                raw_violations += sum(
                    (c.mode == "disabled" and c.modality in selected.raw_modalities)
                    or (
                        c.mode == "required"
                        and c.modality not in selected.raw_modalities
                    )
                    for c in session.request.controls
                )
            rows.append(
                dict(
                    index=i,
                    request=session.request.to_dict(),
                    target_operation=OPERATIONS[target_op],
                    target_modalities=[
                        m
                        for m, active in zip(prediction.modalities, target_mask)
                        if active
                    ],
                    raw_operation=selected.raw_operation,
                    raw_modalities=list(selected.raw_modalities),
                    enforced_operation=selected.operation,
                    enforced_requests=[asdict(r) for r in selected.requests],
                    operation_logits=prediction.operation_logits[j].cpu().tolist(),
                    modality_logits=prediction.modality_logits[j].cpu().tolist(),
                    completion_probability=selected.completion_probability,
                    clarification_probability=selected.clarification_probability,
                    operation_error=int(
                        selected.raw_operation != OPERATIONS[target_op]
                    ),
                    modality_errors=[
                        int(a != b) for a, b in zip(raw_mask, target_mask)
                    ],
                    completion_error=int(
                        (selected.completion_probability >= 0.5)
                        != (OPERATIONS[target_op] == "finish")
                    ),
                    shuffled_operation_error=int(
                        int(shuffled.operation_logits[j].argmax()) != target_op
                    ),
                    shuffled_completion_error=int(
                        (float(shuffled.completion_logits[j]) >= 0)
                        != (target_op == OPERATIONS.index("finish"))
                    ),
                    shuffled_modality_errors=(
                        (shuffled.modality_logits[j] >= 0)
                        != data.modality_targets[i].to(state.tokens.device).bool()
                    )
                    .int()
                    .cpu()
                    .tolist(),
                    lexical_operation_error=int(
                        int(ref_operations[neighbor]) != target_op
                    ),
                    lexical_modality_errors=(
                        ref_modalities[neighbor].bool()
                        != data.modality_targets[i].bool()
                    )
                    .int()
                    .tolist(),
                    lexical_completion_error=int(
                        (int(ref_operations[neighbor]) == OPERATIONS.index("finish"))
                        != (target_op == OPERATIONS.index("finish"))
                    ),
                    gate_violations=violations,
                    raw_gate_violations=raw_violations,
                )
            )
    metrics = {
        "task_" + k: float(np.mean([r[k] for r in rows]))
        for k in (
            "operation_error",
            "completion_error",
            "shuffled_operation_error",
            "shuffled_completion_error",
            "lexical_operation_error",
            "lexical_completion_error",
        )
    }
    for prefix in ("", "shuffled_", "lexical_"):
        metrics["task_" + prefix + "modality_error"] = float(
            np.mean([r[prefix + "modality_errors"] for r in rows])
        )
    for j, name in enumerate(model.task_policy.modalities):
        emit_rows = [r for r in rows if r["target_operation"] == "emit"]
        metrics[f"task_{name}_error"] = float(
            np.mean([r["modality_errors"][j] for r in rows])
        )
        if emit_rows:
            metrics[f"task_emit_{name}_error"] = float(
                np.mean([r["modality_errors"][j] for r in emit_rows])
            )
    metrics["task_gate_violations"] = sum(r["gate_violations"] for r in rows)
    metrics["task_raw_gate_violations"] = sum(r["raw_gate_violations"] for r in rows)
    return metrics, rows


class RealEpisodes:
    def __init__(self, root, split, count, image_size, history, horizon):
        self.data = PushTSequences(
            root, split, history=history, horizon=horizon, limit=count
        )
        self.image_size = image_size
        self.history, self.horizon = history, horizon
        self.identity = dict(
            self.data.identity,
            resize=image_size,
            modalities=["image", "video"],
            action_transform="2 * normalized_xy - 1",
            time_unit="one recorded transition; no physical timestamp calibration",
        )

    def __len__(self):
        return len(self.data)

    def batch(self, indices, device="cpu"):
        raw = self.data.batch(indices, device)
        images = torch.cat((raw["history"], raw["future"]), 1)
        b, t = images.shape[:2]
        images = F.interpolate(
            images.flatten(0, 1),
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 3, self.image_size, self.image_size)
        actions = torch.cat((raw["history_actions"], raw["actions"]), 1) * 2 - 1
        if not torch.isfinite(actions).all() or (actions.abs() > 1.00001).any():
            raise ValueError(
                "Prepared action coordinates do not match normalized [0,1] contract"
            )
        sources = []
        for index in indices:
            episode, start = self.data.windows[int(index)]
            group = int(self.data.episodes[episode]["group_id"])
            sources.append(
                f"pusht/{self.data.identity['split']}/group-{group}/start-{start}"
            )
        return {"images": images, "actions": actions, "sources": sources}


def observations(batch, t):
    images = batch["images"]
    times = images.new_full((len(images), 1), t)
    start = max(0, t - 1)
    video_times = torch.arange(start, t + 1, device=images.device, dtype=images.dtype)[
        None
    ].expand(len(images), -1)
    result = {
        "image": Observation(images[:, t : t + 1], times),
        "video": Observation(images[:, start : t + 1], video_times),
    }
    if "audio" in batch:
        result["audio"] = Observation(batch["audio"][:, t : t + 1], times)
    if "text" in batch:
        text = batch["text"][:, t]
        result["text"] = Observation(text, times.expand_as(text), text != 0)
    return result


def observe_history(model, batch, history, *, trace=None, dropout=0.0):
    state = model.initial_state(len(batch["images"]))
    for t in range(history):
        obs = observations(batch, t)
        if dropout:
            obs = {
                k: v
                for k, v in obs.items()
                if k == "image" or torch.rand(()) >= dropout
            }
        state = model.observe(
            state,
            obs,
            time=t,
            previous_action=None if t == 0 else batch["actions"][:, t - 1],
            trace=trace,
        )
        source = json.dumps(batch["sources"], separators=(",", ":")) + f"/frame-{t}"
        state = model.remember(state, source=source)
    return model.think(state, steps=1, trace=trace)


def output_losses(model, state, batch, t):
    prefix = batch["text"][:, t, :-1] if "text" in batch else None
    outputs = model.decode(
        state,
        text_prefix=prefix,
        modalities=[
            "image",
            *(["audio"] if "audio" in batch else []),
            *(["text"] if "text" in batch else []),
        ],
    )
    losses = {
        "image_mse": (outputs["image"] - batch["images"][:, t]).square().mean((1, 2, 3))
    }
    if "audio" in batch:
        losses["audio_mse"] = (outputs["audio"] - batch["audio"][:, t]).square().mean(1)
    if "text" in batch:
        target = batch["text"][:, t, 1:]
        ce = F.cross_entropy(
            outputs["text"].transpose(1, 2), target, ignore_index=0, reduction="none"
        )
        losses["text_ce"] = ce.sum(1) / (target != 0).sum(1).clamp_min(1)
    return losses


class LearningState(nn.Module):
    """Checkpoint the deployed model, its training-only EMA copy and replay scores."""

    def __init__(self, model, population, *, diagnostic=False):
        super().__init__()
        self.agent = model
        self.target = copy.deepcopy(model).requires_grad_(False).eval()
        self.register_buffer("replay_errors", torch.ones(population))
        self.register_buffer("proposals", torch.tensor(0))
        self.register_buffer("accepted", torch.tensor(0))
        if diagnostic:
            self.register_buffer(
                "diagnostic_elapsed_seconds", torch.tensor(0.0, dtype=torch.float64)
            )
        if hasattr(model, "recall_head"):
            # Recall uses the second copy for development-selected weights, not EMA.
            self.register_buffer("best_nll", torch.tensor(1e30, dtype=torch.float64))
            self.register_buffer("best_step", torch.tensor(0))
            self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float64))

    @torch.no_grad()
    def update_target(self, decay):
        for target, current in zip(self.target.parameters(), self.agent.parameters()):
            target.lerp_(current, 1 - decay)
        for target, current in zip(self.target.buffers(), self.agent.buffers()):
            target.copy_(current)


def belief_likelihoods(model, state, batch, t, *, tokens=None):
    """Explicit adapter distributions, joint within a modality; no cross-modal independence claim."""
    from pathwm.training.belief import gaussian_log_likelihood

    values = state.tokens if tokens is None else tokens
    result, dimensions = {}, {}
    for name, key in (("image", "images"), ("audio", "audio")):
        if key in batch:
            target = batch[key][:, t]
            prediction = model.decoders[name](values)
            result[name] = gaussian_log_likelihood(prediction, target, sigma=0.1)
            dimensions[name] = target[0].numel()
    if "text" in batch:
        text = batch["text"][:, t]
        prediction = model.decoders["text"](values, text[:, :-1])
        ce = F.cross_entropy(
            prediction.transpose(1, 2), text[:, 1:], ignore_index=0, reduction="none"
        )
        result["text"] = -ce.sum(1)
        dimensions["text"] = (text[:, 1:] != 0).sum(1).clamp_min(1)
    return result, dimensions


def belief_objective(learner, batch, history, horizon, dropout):
    """Bounded replay; two-draw marginal NLL with biased straight-through gradients.

    Missingness is random item availability, independent of hidden values. The same
    frame mask applies to duplicated image/video views. Full teachers share only
    the partial causal prefix and never commit their current observation.
    """
    from pathwm.models.belief_state import Packet
    from pathwm.models.hybrid_memory import detached
    from pathwm.models.modalities import position
    from pathwm.training.belief import split_kl, partial_kl, marginal_nll, marking_loss

    model = learner.agent
    state = model.initial_state(len(batch["images"]))
    dynamics_losses, representation_losses, partial_losses, evidence_losses = (
        [],
        [],
        [],
        [],
    )
    labelled = []
    first_record, first_state = None, None
    for t in range(history):
        pending = model.begin_event(
            state,
            event_id=f"event-{t}",
            ordinal=t,
            time=t,
            action=None if t == 0 else batch["actions"][:, t - 1],
            replay=True,
        )
        full = observations(batch, t)
        visible_frames = (
            torch.rand(len(state.tokens), t + 1, device=state.tokens.device) >= dropout
        )
        packets, full_packets = [], []
        for name, observation in sorted(full.items()):
            valid = (
                observation.valid
                if observation.valid is not None
                else torch.ones_like(observation.times, dtype=torch.bool)
            )
            if name in ("image", "video"):
                keep = visible_frames.gather(1, observation.times.long())
            else:
                keep = torch.rand_like(observation.times) >= dropout
            partial = replace(observation, valid=valid & keep)
            packets.append(Packet(f"{t}/{name}", name, partial))
            full_packets.append(Packet(f"{t}/{name}", name, observation))
        # A complete known packet set needs only one correction, independent of arrival order.
        partial = model.correct_packets(replace(pending, packets=tuple(packets)))
        with torch.no_grad():
            teacher = learner.target.correct_packets(
                replace(pending, packets=tuple(full_packets))
            )
        current = partial.state
        valid = current.evidence_valid
        dyn, rep = split_kl(current.logits, current.prior_logits, valid=valid)
        dynamics_losses.append(dyn * valid.sum())
        representation_losses.append(rep * valid.sum())
        labelled.append(valid.sum())
        partial_losses.append(partial_kl(teacher.state.logits, current.logits))
        # Source features have their own grounding loss, with no inferred-state input.
        source_ll, dims = belief_likelihoods(
            model, current, batch, t, tokens=current.evidence
        )
        evidence_losses.append(
            sum(
                ((-value / dims[k]) * valid).sum() / valid.sum().clamp_min(1)
                for k, value in source_ll.items()
            )
        )
        state = model.commit_event(partial)
        if t == 0 and state.memory is not None:
            first_record, first_state = state.memory.recent[-1], detached(state)
    state = model.think(state, steps=1)
    denominator = torch.stack(labelled).sum().clamp_min(1)
    weighted = dict(
        dynamics_kl=torch.stack(dynamics_losses).sum() / denominator,
        representation_kl=0.1 * torch.stack(representation_losses).sum() / denominator,
        partial_kl=0.1 * torch.stack(partial_losses).mean(),
        evidence_nll=0.05 * torch.stack(evidence_losses).mean(),
    )
    starts = model.sample_beliefs(state, 2)
    current = [
        belief_likelihoods(model, branch, batch, history - 1) for branch in starts
    ]
    for name in current[0][0]:
        weight = 0.1 if name == "text" else 1.0
        weighted[f"reconstruct_{name}_nll"] = (
            0.25
            * weight
            * (
                marginal_nll(torch.stack([v[0][name] for v in current]))
                / current[0][1][name]
            ).mean()
        )
    futures, observable_errors = starts, {}
    for h in range(horizon):
        t = history + h
        futures = [
            model.imagine(branch, batch["actions"][:, t - 1], dt=1.0, sample=True)
            for branch in futures
        ]
        predictions = [
            belief_likelihoods(model, branch, batch, t) for branch in futures
        ]
        for name in predictions[0][0]:
            weight = 0.1 if name == "text" else 1.0
            loss = (
                marginal_nll(torch.stack([v[0][name] for v in predictions]))
                / predictions[0][1][name]
            ).mean()
            key = f"future_{name}_nll"
            weighted[key] = weighted.get(key, 0) + weight * loss / horizon
        measurements = [output_losses(model, branch, batch, t) for branch in futures]
        for name in measurements[0]:
            observable_errors.setdefault(name, []).append(
                torch.stack([v[name] for v in measurements]).mean(0)
            )
    error = torch.stack(observable_errors["image_mse"]).mean(0)
    mean, scale = model.action_head(state.tokens)
    action = batch["actions"][:, history - 1].clamp(-0.9999, 0.9999)
    weighted["action_nll"] = (
        0.05
        * (
            0.5 * ((torch.atanh(action) - mean) * (-scale).exp()).square()
            + scale
            + 0.5 * np.log(2 * np.pi)
            + torch.log1p(-action.square())
        ).mean()
    )
    weighted["self_error_mse"] = (
        0.1 * (model.monitor(state.tokens) - error.detach()).square().mean()
    )

    # Query an old observation after exact recency eviction, without the live belief
    # as a shortcut. The adapter supplies the requested event time, not its content.
    if (
        first_record is not None
        and history >= model.memory.capacity + model.memory.block_size
        and first_record.event_id
        not in {r.event_id for r in state.memory.recent + state.memory.staging}
    ):
        query = model.initial[None, model.layout["world"]].expand(
            len(state.tokens), -1, -1
        )
        query = (
            query
            + position(state.time.new_zeros(len(query)), model.width).to(query.dtype)[
                :, None
            ]
        )

        def recall_loss(bank):
            recalled = query + model.memory.read(
                replace(state, memory=bank), query=query, consumer="thinking"
            )
            ll, dims = belief_likelihoods(model, state, batch, 0, tokens=recalled)
            return -ll["image"] / dims["image"]

        without = recall_loss(state.memory)
        protected = state.memory.protected
        # One reserved comparison slot, same capacity. With existing marks, compare
        # replacing the lowest agent mark; user marks never provide a free extra slot.
        candidates = [
            (r.score, i) for i, r in enumerate(protected) if r.author != "user"
        ]
        room = len(protected) < model.memory.protected_capacity
        if room or candidates:
            selected = list(protected)
            if not room:
                selected.pop(min(candidates)[1])
            selected.append(
                replace(first_record, author="agent", detail="training candidate")
            )
            with_detail = recall_loss(replace(state.memory, protected=tuple(selected)))
            score = model.memory.propose_mark_score(first_record, first_state.tokens)
            weighted["marking_mse"] = 0.05 * marking_loss(score, without, with_detail)
            weighted["protected_recall_nll"] = 0.05 * with_detail.mean()
        weighted["delayed_recall_nll"] = 0.1 * without.mean()

    # Fixed-reader distribution distillation into separately parameterized view
    # compressors. The target and query are detached; reader parameters are frozen
    # functionally, so this auxiliary cannot make a moving reader hide information loss.
    if state.memory is not None and len(state.memory.recent) >= 2:
        records = detached(state.memory.recent[-2:])
        compressed = model.memory.summarize(records)
        consolidated = model.memory.consolidate(detached(compressed), None)
        query = state.h.detach()
        losses = []
        for reader in model.memory.readers.values():
            params = {k: v.detach() for k, v in reader.named_parameters()}

            def read(groups):
                return torch.func.functional_call(
                    reader, params, (query, groups, state.time)
                )

            with torch.no_grad():
                target = read(((), records, ())).flatten(1).softmax(-1)
            for record in (compressed, consolidated):
                log_prediction = read(((), (record,), ())).flatten(1).log_softmax(-1)
                losses.append(F.kl_div(log_prediction, target, reduction="batchmean"))
        weighted["compression_read_kl"] = 0.1 * torch.stack(losses).mean()
    diagnostics = dict(
        **{k: torch.stack(v).detach().mean() for k, v in observable_errors.items()},
        categorical_entropy=-(state.logits.exp() * state.logits)
        .sum(-1)
        .mean()
        .detach(),
        dynamics_kl=weighted["dynamics_kl"].detach(),
        memory_tensor_bytes=state.tokens.new_tensor(
            model.memory.storage_bytes(state.memory)
        ),
    )
    if "tasks" in batch:
        losses, raw = task_losses(model, state, batch)
        weighted.update(losses)
        diagnostics.update({k: v.detach() for k, v in raw.items()})
    return weighted, error.detach(), diagnostics


def objective(learner, batch, history=2, horizon=2, dropout=0.0):
    if "entity_inputs" in batch:
        logits = learner.agent(batch["entity_inputs"])
        terms = [
            -(p * score.log_softmax(-1)).sum(-1)
            for score, p in zip(logits, batch["entity_targets"])
        ]
        return (
            {n + "_nll": v.mean() / len(terms) for n, v in zip(ENTITY_NAMES, terms)},
            torch.stack(terms).mean(0).detach(),
            {},
        )
    if "fact_observation" in batch:
        entity, location = learner.agent(batch["fact_observation"])
        entity_loss = F.cross_entropy(entity, batch["entities"], reduction="none")
        location_loss = F.cross_entropy(location, batch["locations"], reduction="none")
        losses = dict(
            entity_nll=0.5 * entity_loss.mean(), location_nll=0.5 * location_loss.mean()
        )
        return losses, (0.5 * (entity_loss + location_loss)).detach(), {}
    if "recall_inputs" in batch:
        return recall_objective(learner, batch)
    if isinstance(learner.agent, BeliefAgent):
        return belief_objective(learner, batch, history, horizon, dropout)
    model = learner.agent
    state = observe_history(model, batch, history, dropout=dropout)
    reconstruction = output_losses(model, state, batch, history - 1)
    with torch.no_grad():
        target = observe_history(learner.target, batch, history)
    totals, errors, latent_losses = {}, [], []
    future = state
    for h in range(horizon):
        t = history + h
        future = model.imagine(future, batch["actions"][:, t - 1], dt=1.0)
        losses = output_losses(model, future, batch, t)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0) + v / horizon
        errors.append(losses["image_mse"])
        with torch.no_grad():
            target = learner.target.observe(
                target,
                observations(batch, t),
                time=t,
                previous_action=batch["actions"][:, t - 1],
            )
            target = learner.target.think(target, steps=1)
        residual = (future.tokens - target.tokens.detach()) * (-future.log_scale).exp()
        latent_losses.append(
            (
                0.5 * residual.square() + future.log_scale + 0.5 * np.log(2 * np.pi)
            ).mean()
        )
    image_error = torch.stack(errors).mean(0)
    action = batch["actions"][:, history - 1].clamp(-0.9999, 0.9999)
    action_mean, action_scale = model.action_head(state.tokens)
    raw = torch.atanh(action)
    action_nll = (
        0.5 * ((raw - action_mean) * (-action_scale).exp()).square()
        + action_scale
        + 0.5 * np.log(2 * np.pi)
        + torch.log1p(-action.square())
    ).mean()
    # Explicit small weights; data-space objectives anchor the evolving latent target.
    weighted = {
        f"future_{k}": v.mean() * (0.1 if k == "text_ce" else 1.0)
        for k, v in totals.items()
    }
    weighted.update(
        {
            f"reconstruct_{k}": 0.25 * v.mean() * (0.1 if k == "text_ce" else 1.0)
            for k, v in reconstruction.items()
        }
    )
    weighted["latent_nll"] = 0.01 * torch.stack(latent_losses).mean()
    weighted["action_nll"] = 0.05 * action_nll
    weighted["self_error_mse"] = (
        0.1 * (model.monitor(state.tokens) - image_error.detach()).square().mean()
    )
    std = state.tokens.flatten(0, 1).std(0, unbiased=False)
    weighted["variance_floor"] = 0.01 * F.relu(0.1 - std).mean()
    diagnostics = {k: v.detach().mean() for k, v in totals.items()}
    diagnostics.update(
        latent_error_mse=(future.tokens - target.tokens).detach().square().mean(),
        latent_predicted_variance=future.log_scale.detach().mul(2).exp().mean(),
        self_error_mse=(model.monitor(state.tokens) - image_error.detach())
        .detach()
        .square()
        .mean(),
    )
    if "tasks" in batch:
        losses, task_raw = task_losses(model, state, batch)
        weighted.update(losses)
        diagnostics.update({k: v.detach() for k, v in task_raw.items()})
    return weighted, image_error.detach(), diagnostics


def update(learner, optimizer, data, indices, settings):
    training_mode(learner)
    batch = data.batch(indices, settings["device"])
    optimizer.zero_grad(set_to_none=True)
    losses, errors, _ = objective(
        learner, batch, settings["history"], settings["horizon"], dropout=0.25
    )
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite multimodal objective")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters(learner), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    if not isinstance(data, (RecallEpisodes, FactExamples, EntityEpisodes)):
        learner.update_target(settings["ema_decay"])
    with torch.no_grad():
        # Duplicate sampled indices are deliberately averaged, independently of order.
        ids = torch.as_tensor(indices, device=errors.device)
        for index in ids.unique():
            learner.replay_errors[index] = errors[ids == index].mean()
    return {
        **{k: float(v.detach()) for k, v in losses.items()},
        "loss": float(loss.detach()),
        "grad_norm": float(norm),
    }


@torch.no_grad()
def evaluate(learner, data, settings):
    if isinstance(data, RecallEpisodes):
        logits, labels, _ = recall_predictions(learner.agent, data, settings)
        metrics = recall_metrics(
            logits, labels, abstain_cost=settings.get("abstain_cost", 0.25)
        )
        return dict(loss=metrics["nll"], **metrics)
    sums, seen, pooled = {}, 0, []
    with evaluation_mode(learner):
        for start in range(0, len(data), settings["batch_size"]):
            ids = np.arange(start, min(start + settings["batch_size"], len(data)))
            batch = data.batch(ids, settings["device"])
            losses, _, raw = objective(
                learner, batch, settings["history"], settings["horizon"]
            )
            state = observe_history(learner.agent, batch, settings["history"])
            pooled.append(state.tokens.mean(1))
            current = batch["images"][:, settings["history"] - 1]
            target = batch["images"][
                :, settings["history"] : settings["history"] + settings["horizon"]
            ]
            raw["copy_image_mse"] = (current[:, None] - target).square().mean()
            raw["loss"] = sum(losses.values())
            if "audio" in batch:
                raw["silence_audio_mse"] = (
                    batch["audio"][:, settings["history"] :].square().mean()
                )
            for k, v in raw.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(ids)
            seen += len(ids)
    values = torch.cat(pooled)
    centered = values - values.mean(0)
    singular = torch.linalg.svdvals(centered)
    proportions = singular.square() / singular.square().sum().clamp_min(1e-12)
    rank = (
        (-(proportions * proportions.clamp_min(1e-12).log()).sum()).exp()
        if singular.square().sum() > 1e-12
        else 0.0
    )
    result = {
        **{k: v / seen for k, v in sums.items()},
        "evaluated_windows": seen,
        "latent_between_example_std": float(values.std(0, unbiased=False).mean()),
        "latent_effective_rank": float(rank),
    }
    if isinstance(data, InstructionEpisodes):
        with evaluation_mode(learner):
            task_metrics, _ = task_evaluation(learner.agent, data, settings)
        result.update(task_metrics)
    return result


def make_data(settings, split):
    if settings["dataset"] == "entity-matching":
        dataset = (
            VariableEntityMatches
            if settings.get("entity_variable", False)
            else EntityMatches
        )
        return dataset(
            split, settings.get(f"{split}_windows", 512 if split == "train" else 256)
        )
    if settings["dataset"] in ("entities", "entity-matching"):
        return EntityEpisodes(
            split,
            settings.get(f"{split}_windows", 512 if split == "train" else 256),
            noise=settings.get("entity_noise", 0.0),
        )
    if settings["dataset"] == "facts":
        return FactExamples(
            split, settings.get(f"{split}_windows", 96 if split == "train" else 32)
        )
    if settings["dataset"] == "recall":
        return RecallEpisodes(
            split=split,
            count=settings.get(f"{split}_windows", 15),
            history=settings["history"],
            recent=settings.get("memory_recent", 32),
            block=settings.get("memory_block", 8),
            blocks=settings.get("memory_blocks", 16),
            truncate=settings.get("recall_truncate", 32),
            abstain_cost=settings.get("abstain_cost", 0.25),
            mode=settings.get("recall_mode", "history"),
        )
    count = (
        settings["train_windows"]
        if split == "train"
        else settings["validation_windows"]
    )
    if settings["dataset"] in ("synthetic", "instructions"):
        factory = (
            InstructionEpisodes
            if settings["dataset"] == "instructions"
            else SyntheticEpisodes
        )
        return factory(
            split=split,
            count=count,
            image_size=settings["image_size"],
            audio_samples=settings["audio_samples"],
            history=settings["history"],
            horizon=settings["horizon"],
        )
    return RealEpisodes(
        settings["data_root"],
        split,
        count,
        settings["image_size"],
        settings["history"],
        settings["horizon"],
    )


def check(settings):
    seed_everything(settings["seed"])
    data = make_data(settings, "train")
    learner = LearningState(
        build_model(
            settings["width"],
            settings["image_size"],
            settings["audio_samples"],
            state_model=settings.get("state_model", "gaussian"),
            memory_recent=settings.get("memory_recent", 32),
            memory_block=settings.get("memory_block", 8),
            memory_blocks=settings.get("memory_blocks", 16),
            recall=settings["dataset"] == "recall",
            facts=settings["dataset"] == "facts",
            entities=settings["dataset"] in ("entities", "entity-matching"),
            entity_matching=settings["dataset"] == "entity-matching",
            entity_reader=settings.get("entity_reader", "recurrent"),
            entity_association=settings.get("entity_association", "raw"),
            fact_reader=settings.get("fact_reader", "direct"),
            fact_encoder_weights=settings.get("fact_encoder_weights"),
        ),
        len(data),
    ).to(settings["device"])
    batch = data.batch(
        np.arange(min(settings["batch_size"], len(data))), settings["device"]
    )
    losses, _, _ = objective(learner, batch, settings["history"], settings["horizon"])
    sum(losses.values()).backward()
    return {
        "data": data.identity,
        "encoder_initialization": getattr(
            learner.agent, "encoder_initialization", None
        ),
        "parameters": sum(p.numel() for p in learner.agent.parameters()),
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "gradients": {
            name: any(
                p.grad is not None and bool(p.grad.abs().sum())
                for p in module.parameters()
            )
            for name, module in learner.agent.named_children()
        },
        "target_has_gradients": any(
            p.grad is not None for p in learner.target.parameters()
        ),
    }


@torch.no_grad()
def save_task_example(run, model, data, settings):
    """Expose actual proposals alongside a caller-requested emission/loopback."""
    batch = data.batch([0], settings["device"])
    state = observe_history(model, batch, settings["history"])
    user, agent = Actor("user", "demo-user"), Actor("agent", "model")
    session = TaskSession(
        TaskRequest(
            "demo",
            "Respond with image.",
            user,
            (
                OutputControl("image", "required", user),
                OutputControl("audio", "disabled", user),
                OutputControl("text", "disabled", user),
                OutputControl("video", "disabled", user),
            ),
        )
    )
    trace = {}
    selection = model.decide(state, session, agent=agent, trace=trace)
    # Show the learned decision honestly even when it does not choose to emit.
    # A separate explicit user request exercises generation regardless of weights.
    requests = (
        selection.requests
        if selection.operation == "emit"
        else (OutputRequest("demo/user-image", "image", user, "demo"),)
    )
    emission = model.emit(state, session, requests, produced_by=agent, trace=trace)
    if emission.errors:
        raise RuntimeError(f"Task example generation failed: {emission.errors}")
    output = emission.outputs[0]
    reflected = model.reflect(
        state, {"image": output.loopback(state.time)}, trace=trace
    )
    record = {
        "selection": asdict(selection),
        "emission_trigger": "learned selection"
        if selection.operation == "emit"
        else "explicit user request; learned decision shown separately",
        "outputs": [o.provenance.to_dict() for o in emission.outputs],
        "remaining": list(emission.session.remaining),
        "world_time_before": state.time.tolist(),
        "world_time_after": reflected.time.tolist(),
        "observed_count_before": state.observation_count,
        "observed_count_after": reflected.observation_count,
        "reflection_ancestry": [p.to_dict() for p in reflected.generated_ancestry],
        "caveat": "Control/provenance demonstration; generated content is not validated instruction following.",
    }
    torch.save(
        {
            "before": state.to_dict(),
            "after": reflected.to_dict(),
            "task": emission.session.to_dict(),
            "trace": trace,
        },
        run.path / "task_inspection.pt",
    )
    Image.fromarray(
        (output.values[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(
            "uint8"
        )
    ).save(run.path / "task_output.png")
    if isinstance(data, InstructionEpisodes):
        metrics, rows = task_evaluation(model, data, settings)
        atomic_json(
            run.path / "task_decisions.json", {"metrics": metrics, "examples": rows}
        )
        record["evaluation"] = metrics
    atomic_json(run.path / "task_demo.json", record)
    return record


@torch.no_grad()
def save_examples(run, learner, data, settings):
    if settings["dataset"] in ("entities", "entity-matching"):
        render_report(run.path)
        return
    if isinstance(data, (RecallEpisodes, FactExamples, EntityEpisodes)):
        return write_report(run.path)
    with evaluation_mode(learner):
        model = learner.agent
        batch = data.batch(np.arange(min(3, len(data))), settings["device"])
        trace = {}
        state = observe_history(model, batch, settings["history"], trace=trace)
        if "tasks" in batch:
            model.task_predictions(state, batch["tasks"], trace=trace)
        imagined, future = [], state
        for h in range(settings["horizon"]):
            step_trace = {}
            future = model.imagine(
                future,
                batch["actions"][:, settings["history"] - 1 + h],
                trace=step_trace,
            )
            imagined.append(future)
            trace.update({f"future.{h}.{k}": v for k, v in step_trace.items()})
        video = model.decode_video(imagined)
        audio = torch.stack(
            [model.decode(s, modalities=["audio"])["audio"] for s in imagined], 1
        )
        text = model.generate_text(imagined[-1], max_tokens=24)
        proposed = model.propose_action(state)
        # Compare proposal, zero and negative proposal under an explicit goal image.
        candidates = torch.stack((proposed, torch.zeros_like(proposed), -proposed), 1)
        candidates = candidates[:, :, None].expand(-1, -1, settings["horizon"], -1)
        goal = batch["images"][:, -1]
        decision = plan(
            model,
            state,
            candidates,
            lambda s: (
                (model.decode(s, modalities=["image"])["image"] - goal)
                .square()
                .mean((1, 2, 3))
            ),
            lower=-1.0,
            upper=1.0,
            trace=trace,
        )
        baseline_image = model.decode(state, modalities=["image"])["image"]
        ablations = {}
        for role in model.layout:
            if isinstance(model, BeliefAgent) and role == "world":
                continue
            changed = model.decode(
                model.intervene(state, role, 0.0), modalities=["image"]
            )["image"]
            ablations[role] = float((changed - baseline_image).square().mean())
        torch.save(
            {
                "state": state.to_dict(),
                "future": [s.to_dict() for s in imagined],
                "trace": trace,
            },
            run.path / "inspection.pt",
        )
        np.savez_compressed(
            run.path / "inspection.npz",
            attention=trace["observe.attention"][0].mean(0).numpy(),
            token_activity=state.tokens[0].abs().cpu().numpy(),
            video=video.cpu().numpy(),
            audio=audio.cpu().numpy(),
        )
        np.savez_compressed(
            run.path / "multimodal_outputs.npz",
            video=video.cpu().numpy(),
            audio=audio.cpu().numpy(),
            text_tokens=text.cpu().numpy(),
            actions=decision.actions.cpu().numpy(),
        )
        frames = [
            Image.fromarray(
                (frame.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            )
            for frame in video[0]
        ]
        frames[0].save(
            run.path / "imagined.gif",
            save_all=True,
            append_images=frames[1:],
            duration=250,
            loop=0,
        )
        with wave.open(str(run.path / "imagined.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            # Clips are contiguous decoder waveform chunks, not claimed synchronized speech.
            stream.writeframes(
                (audio[0].flatten().cpu().clamp(-1, 1).numpy() * 32767)
                .astype("<i2")
                .tobytes()
            )
        atomic_json(
            run.path / "inspection.json",
            {
                "step": run.step,
                "latent_groups": model.groups,
                "generated_text": [bytes_text(row) for row in text],
                "plan_scores": decision.scores.cpu().tolist(),
                "selected_candidates": decision.indices.cpu().tolist(),
                "memory_sources": list(state.memory.sources),
                "time": state.time.cpu().tolist(),
                "future_times": [s.time.cpu().tolist() for s in imagined],
                "self_error_estimate": model.monitor(state.tokens).cpu().tolist(),
                "group_activity_rms": {
                    name: float(state.tokens[:, sl].square().mean().sqrt())
                    for name, sl in model.layout.items()
                },
                "zero_group_image_change_mse": ablations,
                "attention_inputs": trace["observe.input_modalities"],
                "attention_input_scales": trace["observe.input_scales"],
                "feature_code": trace["encode.feature_code"].tolist()
                if "encode.feature_code" in trace
                else None,
                "feature_code_source": trace.get(
                    "encode.condition_source", "unconditioned_source_evidence"
                ),
                "feature_code_time": trace["encode.condition_time"].tolist()
                if "encode.condition_time" in trace
                else None,
                "state_model": settings.get("state_model", "gaussian"),
                "memory_tensor_bytes": model.memory.storage_bytes(state.memory)
                if isinstance(model, BeliefAgent)
                else None,
                "audio_sample_rate": 8000,
                "audio_samples_per_state": settings["audio_samples"],
                "trace_keys": sorted(trace),
                "task_demo": save_task_example(run, model, data, settings),
                "caveats": [
                    "Development outputs, not trained semantic or physical capabilities.",
                    "The demonstration planner is given the true final image as its explicit goal.",
                    "The goal is used only by scoring; future observations never enter imagined inference.",
                    "Uncertainty is conditional latent spread, not a calibrated confidence score.",
                    "Audio/text outputs on PushT are untrained because those modalities are absent.",
                    "Token roles, attention weights and PCA colors are not causal explanations.",
                ],
            },
        )
        features = {
            name: state.tokens[:, sl].transpose(1, 2).unsqueeze(2)
            for name, sl in model.layout.items()
        }
        if "task.tokens" in trace:
            features["task_tokens"] = trace["task.tokens"].transpose(1, 2).unsqueeze(2)
        # Reuse captured processed inputs for the existing per-feature PCA renderer.
        for key, value in trace.items():
            if (
                key.startswith("encode.")
                and ".scale." in key
                and key.endswith(".values")
            ):
                prefix = key.removesuffix(".values")
                grid = trace[prefix + ".grid"]
                features[prefix.removeprefix("encode.")] = value.transpose(
                    1, 2
                ).reshape(
                    len(value), value.shape[-1], int(np.prod(grid[:-1])), grid[-1]
                )
        write_report(
            run.path, {"rgb": batch["images"][:, -1]}, {"rgb": video[:, -1]}, features
        )


def train(settings, output, *, resume=False, stop_after=None):
    is_recall = settings["dataset"] == "recall"
    is_facts = settings["dataset"] == "facts"
    is_entities = settings["dataset"] in ("entities", "entity-matching")
    diagnostic = (
        is_entities
        or is_facts
        or (is_recall and settings.get("recall_mode") == "current-recent")
    )
    if (is_facts or is_entities) and (
        settings["improve_every"] or settings["device"] != "cpu"
    ):
        raise ValueError("Fact diagnostic requires CPU and no extra-update gate")
    if diagnostic and (
        not np.isfinite(settings.get("max_seconds", 900.0))
        or settings.get("max_seconds", 900.0) <= 0
    ):
        raise ValueError("Diagnostic requires a positive finite active-time budget")
    if is_recall and (
        settings.get("state_model") != "belief" or settings["improve_every"]
    ):
        raise ValueError(
            "Recall requires belief state and no extra-update admission gate"
        )
    seed_everything(settings["seed"])
    training, validation = (
        make_data(settings, "train"),
        make_data(settings, "validation"),
    )
    learner = LearningState(
        build_model(
            settings["width"],
            settings["image_size"],
            settings["audio_samples"],
            state_model=settings.get("state_model", "gaussian"),
            memory_recent=settings.get("memory_recent", 32),
            memory_block=settings.get("memory_block", 8),
            memory_blocks=settings.get("memory_blocks", 16),
            recall=is_recall,
            facts=is_facts,
            entities=is_entities,
            entity_matching=settings["dataset"] == "entity-matching",
            entity_reader=settings.get("entity_reader", "recurrent"),
            entity_association=settings.get("entity_association", "raw"),
            fact_reader=settings.get("fact_reader", "direct"),
            fact_encoder_weights=settings.get("fact_encoder_weights"),
        ),
        len(training),
        diagnostic=diagnostic,
    ).to(settings["device"])
    initialization = getattr(learner.agent, "encoder_initialization", None)
    if initialization is not None:
        settings = dict(settings, fact_encoder_initialization=initialization)
    optimizer = torch.optim.AdamW(
        trainable_parameters(learner), lr=settings["learning_rate"]
    )
    identities = {"train": training.identity, "validation": validation.identity}
    if diagnostic:
        if set(training.identity["episode_sha256"]) & set(
            validation.identity["episode_sha256"]
        ):
            raise ValueError("Diagnostic train/development episodes overlap")
    elif is_recall:
        calibration, test = (
            make_data(settings, "calibration"),
            make_data(settings, "test"),
        )
        identities.update(calibration=calibration.identity, test=test.identity)
    run = Run(
        output,
        settings=settings,
        data=identities,
        recipe=__file__,
        model=learner,
        optimizer=optimizer,
        device=settings["device"],
        resume=resume,
    )
    elapsed = float(learner.diagnostic_elapsed_seconds) if diagnostic else 0.0
    if diagnostic and not resume:
        atomic_json(
            run.path
            / (
                "entity_manifest.json"
                if is_entities
                else "fact_manifest.json"
                if is_facts
                else "recall_diagnostic_manifest.json"
            ),
            {
                split: dict(
                    episodes=data.manifest, cohorts=data.cohorts, groups=data.groups
                )
                for split, data in (("train", training), ("development", validation))
            },
        )
    started = perf_counter()
    deadline = (
        started + settings.get("max_seconds", 900.0) - elapsed if diagnostic else None
    )

    def save_progress():
        nonlocal started, deadline
        before = perf_counter()
        if diagnostic:
            learner.diagnostic_elapsed_seconds.fill_(elapsed + before - started)
        run.save()
        if diagnostic:
            overhead = perf_counter() - before
            started += overhead
            deadline += overhead

    def training_probe():
        if is_entities:
            raw = entity_predictions(
                learner.agent, training, settings, deadline=deadline
            )
            measured = entity_metrics(
                raw["logits"], raw["targets"], training.cohorts, training.groups
            )
            view = dict(overall=measured["overall"], groups={})
        elif is_facts:
            view = dict(
                overall=fact_metrics(
                    **fact_predictions(
                        learner.agent, training, settings, deadline=deadline
                    )
                ),
                groups={},
            )
        else:
            logits, labels, examples = recall_predictions(
                learner.agent, training, settings, deadline=deadline
            )
            view = diagnostic_metrics(logits, labels, examples, settings)
        run.log(
            dict(
                step=run.step,
                split="diagnostic_train",
                **view["overall"],
                **{
                    f"group_{g}_{k}": row[k]
                    for g, row in view["groups"].items()
                    for k in ("factual_accuracy", "nll", "examples")
                },
            )
        )
        print(
            f"step {run.step}/{settings['steps']} training factual NLL {view['overall']['nll']:.6f}; accuracy {view['overall']['factual_accuracy']:.1%}",
            flush=True,
        )

    def sample():
        if is_recall or is_facts or is_entities:
            return run.sample(len(training), settings["batch_size"])
        probabilities = replay_probabilities(learner.replay_errors.cpu())
        return torch.multinomial(
            probabilities,
            settings["batch_size"],
            replacement=True,
            generator=run.sampler,
        ).numpy()

    def development():
        val = evaluate(learner, validation, settings)
        if is_recall and val["nll"] < float(learner.best_nll):
            learner.target.load_state_dict(learner.agent.state_dict())
            learner.best_nll.fill_(val["nll"])
            learner.best_step.fill_(run.step)
        return val

    try:
        if not run.rows:
            if diagnostic:
                training_probe()
            else:
                run.log(
                    dict(
                        step=0,
                        split="validation",
                        **development(),
                    )
                )
            save_progress()
        end = (
            min(settings["steps"], run.step + stop_after)
            if stop_after is not None
            else settings["steps"]
        )
        for step in range(run.step + 1, end + 1):
            if diagnostic and perf_counter() >= deadline:
                raise TimeoutError("Diagnostic active-time budget exhausted")
            metrics = update(learner, optimizer, training, sample(), settings)
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    main_examples=step * settings["batch_size"],
                    **metrics,
                )
            )
            if settings["improve_every"] and step % settings["improve_every"] == 0:
                metric_names = [
                    "image_mse",
                    *(
                        ["audio_mse", "text_ce"]
                        if settings["dataset"] in ("synthetic", "instructions")
                        else []
                    ),
                ]

                def admission_metrics():
                    measured = evaluate(learner, validation, settings)
                    return {k: measured[k] for k in metric_names}

                admission = try_improvement(
                    learner,
                    optimizer,
                    lambda: update(learner, optimizer, training, sample(), settings),
                    admission_metrics,
                    primary="image_mse",
                    min_improvement=1e-5,
                    tolerances={
                        k: (
                            0.01
                            if k == "text_ce"
                            else 0.001
                            if k == "audio_mse"
                            else 0.0
                        )
                        for k in metric_names
                    },
                    generators=(run.sampler,),
                )
                learner.proposals.add_(1)
                learner.accepted.add_(int(admission["accepted"]))
                run.log(dict(step=step, split="proposal", **admission))
            if diagnostic:
                if step % settings["evaluate_every"] == 0 and step < settings["steps"]:
                    training_probe()
            elif is_recall or step % settings["evaluate_every"] == 0 or step == end:
                val = development()
                run.log(
                    dict(
                        step=step,
                        split="validation",
                        attempted_extra_updates=int(learner.proposals),
                        accepted_extra_updates=int(learner.accepted),
                        **val,
                    )
                )
                message = (
                    f"development factual NLL {val['nll']:.6f}"
                    if is_recall
                    else f"validation image MSE {val['image_mse']:.6f}; copy {val['copy_image_mse']:.6f}"
                )
                print(f"step {step}/{settings['steps']} {message}", flush=True)
            save_progress()
        result = "completed" if run.step == settings["steps"] else "paused"
        if diagnostic and result == "completed":
            finish_diagnostic = (
                finish_entities
                if is_entities
                else finish_facts
                if is_facts
                else finish_recall_diagnostic
            )
            finish_diagnostic(run, learner, training, validation, settings, deadline)
            save_progress()
        elif is_recall and result == "completed":
            finish_recall(run, learner, calibration, test, settings)
            run.save()
        run.status(result, "pending")
    except BaseException as exc:
        if diagnostic and isinstance(exc, TimeoutError):
            result = "stopped"
            save_progress()
            run.status(result, "pending", str(exc))
        else:
            run.status("failed", "pending", f"{type(exc).__name__}: {exc}")
            try:
                render_report(run.path)
            except Exception as report_error:
                run.status("failed", "failed", f"{exc}; report: {report_error}")
            raise
    try:
        save_examples(run, learner, validation, settings)
    except BaseException as exc:
        run.status(result, "failed", f"{type(exc).__name__}: {exc}")
        raise
    return run.path


def export_diagrams(
    output, *, depth=2, width=32, image_size=16, audio_samples=32, seed=42
):
    """Draw current modules and an executed example; no training or downloads."""
    from pathwm.evaluation.diagrams import CallFlow, architecture, write_diagrams
    from pathwm.io import source_record

    def plan_to_image(model, state, candidates, goal):
        return plan(
            model,
            state,
            candidates,
            lambda future: (
                (model.decode(future, modalities=["image"])["image"] - goal)
                .square()
                .mean((1, 2, 3))
            ),
            lower=-1.0,
            upper=1.0,
        )

    def select_outputs(prediction, session, agent):
        return prediction.select(session, agent)

    def loopback_output(output, time):
        return output.loopback(time)

    seed_everything(seed)
    model = build_model(width, image_size, audio_samples).eval()
    data = SyntheticEpisodes(
        count=1,
        history=2,
        horizon=2,
        image_size=image_size,
        audio_samples=audio_samples,
    )
    batch = data.batch([0])
    flow = CallFlow()
    with torch.no_grad(), evaluation_mode(model):
        inputs = {
            name: flow.input(name + " input", observation)
            for name, observation in observations(batch, 1).items()
        }
        initial = flow.call(model.initial_state, 1)
        observed = flow.call(model.observe, initial, inputs, time=1.0)
        remembered = flow.call(
            model.remember, observed, source="synthetic/train/episode-0/frame-1"
        )
        thought = flow.call(model.think, remembered, steps=1)
        action = flow.call(model.propose_action, thought)
        future = flow.call(model.imagine, thought, action, dt=1.0)
        later = flow.call(model.imagine, future, action, dt=1.0)
        outputs = flow.call(model.decode, future, modalities=["image", "audio"])
        for name, value in outputs.items():
            flow.output(name + " output", value)
        flow.output("text output", flow.call(model.generate_text, future, max_tokens=8))
        flow.output("video output", flow.call(model.decode_video, [future, later]))
        candidates = flow.input("Candidate action sequences", torch.zeros(1, 2, 2, 2))
        goal = flow.input("Goal image", batch["images"][:, -1])
        decision = flow.call(plan_to_image, model, thought, candidates, goal)
        flow.output("Selected action sequence", decision.actions)
        graphs = {"architecture": architecture(model, depth), "data_flow": flow.graph()}
        task_flow = CallFlow()
        task_state = task_flow.input("Observed state", thought)
        user, agent = Actor("user", "diagram-user"), Actor("agent", "model")
        session = task_flow.input(
            "Instruction and explicit output controls",
            TaskSession(
                TaskRequest(
                    "diagram",
                    "Respond with image.",
                    user,
                    (OutputControl("image", "required", user),),
                )
            ),
        )
        task_tokens = task_flow.call(model.task_tokens, task_state, [session])
        prediction = task_flow.call(model.task_policy, task_tokens)
        selection = task_flow.call(select_outputs, prediction, session, agent)
        task_flow.output("Raw and gated proposal", selection)
        explicit = task_flow.input(
            "Explicit user image request (independent of proposal)",
            (OutputRequest("diagram/image", "image", user, "diagram"),),
        )
        emission = task_flow.call(
            model.emit, task_state, session, explicit, produced_by=agent
        )
        loopback = task_flow.call(loopback_output, emission.outputs[0], task_state.time)
        reflected = task_flow.call(model.reflect, task_state, {"image": loopback})
        task_flow.output("Attributed output and fulfillment", emission)
        task_flow.output("Reflection; world clock unchanged", reflected)
        graphs["task_flow"] = task_flow.graph()
        # Capture actual processor/merge calls without duplicating the encoder loop.
        for name, encoder in model.encoders.items():
            if not hasattr(encoder, "pyramid"):
                continue
            scales_flow = CallFlow()
            code = scales_flow.input(
                "Pre-observation feature code", model.propose_feature_code(initial)
            )

            def stem_input(module, args):
                scales_flow.input("Positioned stem features", args[0])

            handle = encoder.pyramid.stages[0].register_forward_pre_hook(stem_input)
            modules = {
                f"scale.{i}": module for i, module in enumerate(encoder.pyramid.stages)
            }
            modules.update(
                {
                    f"merge.{i}": module
                    for i, module in enumerate(encoder.pyramid.merges)
                }
            )
            try:
                with scales_flow.watch(modules):
                    pyramid = encoder(
                        inputs[name], condition=code, condition_time=initial.time
                    )
            finally:
                handle.remove()
            for i, scale in enumerate(pyramid.scales):
                scales_flow.output(f"Processed scale {i} for consumers", scale)
            graph = scales_flow.graph()
            graph.update(
                title=name.title() + " input scales",
                relationship="Each scale finishes processing before the next scale or consumers read it.",
            )
            graphs[name + "_scales"] = graph
    root = Path(__file__).resolve().parents[1]
    provenance = dict(
        seed=seed,
        torch_version=torch.__version__,
        device="cpu",
        depth=depth,
        model=dict(
            width=width,
            image_size=image_size,
            audio_samples=audio_samples,
            code_width=model.feature_controller.code_width,
            input_scales={
                name: len(encoder.pyramid.stages)
                for name, encoder in model.encoders.items()
                if hasattr(encoder, "pyramid")
            },
        ),
        data=data.identity,
        initialization="fresh model; no trained-capability claim",
        calls="one observation, memory write, one thought, two imagined steps and one bounded planning call",
        sources={
            str(Path(p).relative_to(root)): sha
            for p, sha in source_record(__file__, model)["files"].items()
        },
    )
    return write_diagrams(
        output,
        graphs,
        provenance,
    )


def entity_growth(weights, output, resume=False, seed=61):
    """Frozen checkpoint screen, using the ordinary run and report lifecycle."""
    from pathwm.models.entities import EntityMatchReader
    from pathwm.evaluation.entity_growth import growth_inputs, evaluate_growth

    seed_everything(seed)
    weights = Path(weights).resolve()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    if any(name.startswith("agent.") for name in state):
        state = {
            name.removeprefix("agent."): value
            for name, value in state.items()
            if name.startswith("agent.")
        }
    model = EntityMatchReader(width=state["matcher.0.weight"].shape[0])
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    before = state_hash(model)
    families = growth_inputs(seed=seed)
    settings = dict(
        seed=seed,
        entity_growth_seed=seed,
        purpose="diagnostic",
        entity_growth_weights=str(weights),
        donor_sha256=file_hash(weights),
        max_seconds=60,
        threshold=0.75,
        capacities=[1, 2, 4, 8],
        families=32,
        mode="frozen-growth",
    )
    run = Run(
        output,
        settings=settings,
        data={"families": digest(families)},
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    result_path = run.path / "entity_growth.json"
    try:
        if resume:
            results = json.loads(result_path.read_text())
            if results["model_sha256"] != before or results["inputs"] != families:
                raise ValueError("Growth cache identity mismatch")
        else:
            results = evaluate_growth(model, families)
            if state_hash(model) != before:
                raise RuntimeError("Frozen matcher changed")
            results.update(inputs=families, model_sha256=before)
            atomic_json(result_path, results)
            for capacity, scores in results["scores"].items():
                run.log(
                    dict(
                        step=0,
                        split="frozen_growth",
                        capacity=int(capacity),
                        create_accuracy=scores["create"]["accuracy"],
                        revisit_accuracy=scores["revisit"]["accuracy"],
                        overflow_accuracy=scores["overflow"]["accuracy"],
                    )
                )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def train_key_box(
    matcher_weights,
    cell_weights,
    output,
    *,
    steps=256,
    families=16,
    resume=False,
    query_switch=False,
    history_pairs=1,
    eval_seed=2401,
    training_seed=2301,
    stress=False,
    max_seconds=240,
    reference_weights=None,
):
    """Train the entity-to-belief workspace bridge, then execute bounded plans."""
    from pathwm.models.key_box import KeyBoxReader
    from pathwm.models.entity_state import EntityStateCell
    from pathwm.evaluation.key_box import (
        key_training_batch,
        evaluate_key_box,
        key_read_losses,
    )

    if history_pairs not in (1, 4):
        raise ValueError("history_pairs must be 1 or 4")
    seed_everything(training_seed)
    matcher_weights, cell_weights = (
        Path(matcher_weights).resolve(),
        Path(cell_weights).resolve(),
    )
    matcher = EntityMatchReader()
    donor = torch.load(matcher_weights, map_location="cpu", weights_only=True)["model"]
    matcher.load_state_dict(
        {
            k.removeprefix("agent."): v
            for k, v in donor.items()
            if k.startswith("agent.")
        }
    )
    cell = EntityStateCell(preserve_no_information=True)
    cell.load_state_dict(
        torch.load(cell_weights, map_location="cpu", weights_only=True)["model"]
    )
    agent = build_model(
        width=16, state_model="belief", memory_recent=2, memory_block=2, memory_blocks=1
    )
    model = KeyBoxReader(agent, matcher, cell)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )
    donor_hashes = dict(matcher=state_hash(matcher), cell=state_hash(cell))
    run = Run(
        output,
        settings=dict(
            seed=training_seed,
            query_switch=query_switch,
            history_pairs=history_pairs,
            reference_sha256=file_hash(reference_weights)
            if reference_weights
            else None,
            steps=steps,
            families=families,
            lr=0.003,
            purpose="diagnostic",
            matcher_sha256=file_hash(matcher_weights),
            cell_sha256=file_hash(cell_weights),
            max_seconds=max_seconds,
            stress=stress,
            planner="supplied_expectimax4",
        ),
        data=dict(training_seed=training_seed, evaluation_seed=eval_seed, batch=32),
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device="cpu",
        resume=resume,
    )
    path = run.path / "key_box.json"
    started = perf_counter()
    try:
        model.train()
        for step in range(run.step, steps):
            state, latent, target = key_training_batch(model, run.sampler)
            loss = torch.stack(
                key_read_losses(
                    model,
                    state,
                    latent,
                    target,
                    query_switch,
                    history_pairs,
                    float(state.time[0]),
                )
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5
            )
            optimizer.step()
            run.step = step + 1
            run.log(dict(step=run.step, split="train", loss=float(loss.detach())))
            if perf_counter() - started > max_seconds:
                raise TimeoutError("Key-box budget exhausted")
        model.eval()
        if not (resume and path.exists()):
            result = evaluate_key_box(model, families=families, seed=eval_seed)
            if donor_hashes != dict(matcher=state_hash(matcher), cell=state_hash(cell)):
                raise RuntimeError("Entity donor changed")
            if perf_counter() - started > max_seconds:
                raise TimeoutError("Key-box budget exhausted")
            if reference_weights:
                reference = copy.deepcopy(model).eval()
                reference.load_state_dict(
                    torch.load(
                        reference_weights, map_location="cpu", weights_only=True
                    )["model"]
                )
                result["reference"] = evaluate_key_box(
                    reference, families=families, seed=eval_seed
                )
                result["reference"]["model_sha256"] = state_hash(reference)
                if perf_counter() - started > max_seconds:
                    raise TimeoutError("Key-box comparison budget exhausted")
            if stress:
                result["stress"] = evaluate_key_box(
                    model, families=families, seed=eval_seed, stress=True
                )
                if reference_weights:
                    result["reference"]["stress"] = evaluate_key_box(
                        reference, families=families, seed=eval_seed, stress=True
                    )
                result["robustness_passed"] = (
                    result["passed"] and result["stress"]["passed"]
                )
                if perf_counter() - started > max_seconds:
                    raise TimeoutError("Key-box stress budget exhausted")
            result["donor_hashes"] = donor_hashes
            result["model_sha256"] = state_hash(model)
            atomic_json(path, result)
            run.save()
        elif json.loads(path.read_text())["model_sha256"] != state_hash(model):
            raise ValueError("Key-box result cache mismatch")
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def diagnose_entity_sources(source, output, resume=False):
    """Inspect a completed source run; no new samples or policy updates."""
    from pathwm.models.entity_relations import RelationWriteGate
    from pathwm.evaluation.source_choice import diagnose_source_changes

    source = Path(source).resolve()
    raw = source / "entity_source_drift.json"
    model = RelationWriteGate().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(source / "last.pt", map_location="cpu", weights_only=False)["model"]
    )
    run = Run(
        output,
        settings=dict(
            seed=0,
            source_checkpoint_sha256=file_hash(source / "last.pt"),
            purpose="diagnostic",
            source=str(source),
            source_sha256=file_hash(raw),
            max_seconds=30,
        ),
        data=dict(source_sha256=file_hash(raw)),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_source_diagnosis.json"
    try:
        if not resume:
            started = perf_counter()
            result = diagnose_source_changes(json.loads(raw.read_text()))
            if perf_counter() - started > 30:
                raise TimeoutError("Source diagnosis budget exhausted")
            result["source_sha256"] = file_hash(raw)
            atomic_json(path, result)
            for condition, counts in result["summary"].items():
                for category, count in counts.items():
                    run.log(
                        dict(
                            step=0,
                            split="source_diagnosis",
                            condition=condition,
                            category=category,
                            count=count,
                        )
                    )
            run.save()
        elif json.loads(path.read_text())["source_sha256"] != file_hash(raw):
            raise ValueError("Source diagnosis cache mismatch")
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_source_coverage(weights, output, resume=False):
    """Online source values after an unannounced quality swap."""
    from pathwm.models.entity_relations import RelationWriteGate
    from pathwm.evaluation.source_choice import evaluate_source_drift

    weights = Path(weights).resolve()
    model = RelationWriteGate().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True)["model"]
    )
    before = state_hash(model)
    run = Run(
        output,
        settings=dict(
            seed=2101,
            innovation_seed=3101,
            exploration_seed=4101,
            change_z=2,
            acquisition="all_deferred",
            variance_floor=0.0001,
            window=32,
            change_block=32,
            change_threshold=0.15,
            epsilon=0.5,
            feedback_cost=0.005,
            worlds=16,
            pairs=384,
            purpose="diagnostic",
            entity_source_coverage=True,
            entity_gate_weights=str(weights),
            gate_file_sha256=file_hash(weights),
            max_seconds=30,
            costs=[0.05, 0.05],
            correlations=[0.9, 0.0],
            defer_bounds=[0.2, 0.8],
            sigmas=[0.03, 0.15, 0.3, 0.6],
        ),
        data=dict(
            generator="gate_shift_examples",
            pairs=384,
            seeds=[2101, 3101, 4101],
        ),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_source_drift.json"
    started = perf_counter()
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(model):
                raise ValueError("Evidence source cache mismatch")
        else:
            result = evaluate_source_drift(model, seed=2101, change_z=2, coverage=True)
            if state_hash(model) != before:
                raise RuntimeError("Frozen source gate changed")
            if perf_counter() - started > 30:
                raise TimeoutError("Evidence source budget exhausted")
            result["model_sha256"] = before
            atomic_json(path, result)
            for condition, policies in result["summary"].items():
                for name, scores in policies.items():
                    run.log(
                        dict(
                            step=0,
                            split="source_drift",
                            condition=condition,
                            policy=name,
                            utility=scores["utility"],
                            late_utility=scores["late_utility"],
                        )
                    )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_source_drift(weights, output, resume=False):
    """Online source values after an unannounced quality swap."""
    from pathwm.models.entity_relations import RelationWriteGate
    from pathwm.evaluation.source_choice import evaluate_source_uncertainty

    weights = Path(weights).resolve()
    model = RelationWriteGate().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True)["model"]
    )
    before = state_hash(model)
    run = Run(
        output,
        settings=dict(
            seed=2001,
            innovation_seed=3001,
            exploration_seed=4001,
            development_seed=1901,
            development_worlds=8,
            change_z_candidates=[2, 3, 4],
            development_reset_limit=0.125,
            variance_floor=0.0001,
            window=32,
            change_block=32,
            change_threshold=0.15,
            epsilon=0.2,
            feedback_cost=0.005,
            worlds=16,
            pairs=384,
            purpose="diagnostic",
            entity_source_drift=True,
            entity_gate_weights=str(weights),
            gate_file_sha256=file_hash(weights),
            max_seconds=120,
            costs=[0.05, 0.05],
            correlations=[0.9, 0.0],
            defer_bounds=[0.2, 0.8],
            sigmas=[0.03, 0.15, 0.3, 0.6],
        ),
        data=dict(
            generator="gate_shift_examples",
            pairs=384,
            seeds=[1901, 2901, 3901, 2001, 3001, 4001],
        ),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_source_drift.json"
    started = perf_counter()
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(model):
                raise ValueError("Evidence source cache mismatch")
        else:
            result = evaluate_source_uncertainty(model)
            if state_hash(model) != before:
                raise RuntimeError("Frozen source gate changed")
            if perf_counter() - started > 120:
                raise TimeoutError("Evidence source budget exhausted")
            result["model_sha256"] = before
            atomic_json(path, result)
            for condition, policies in result["summary"].items():
                for name, scores in policies.items():
                    run.log(
                        dict(
                            step=0,
                            split="source_drift",
                            condition=condition,
                            policy=name,
                            utility=scores["utility"],
                            late_utility=scores["late_utility"],
                        )
                    )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_source_choice(weights, output, resume=False):
    """Outcome-trained source choice with frozen perception."""
    from pathwm.models.entity_relations import RelationWriteGate
    from pathwm.evaluation.source_choice import evaluate_source_choice

    weights = Path(weights).resolve()
    model = RelationWriteGate().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True)["model"]
    )
    before = state_hash(model)
    run = Run(
        output,
        settings=dict(
            seed=1601,
            innovation_seed=2601,
            worlds=16,
            pairs=384,
            purpose="diagnostic",
            entity_source_choice=True,
            entity_gate_weights=str(weights),
            gate_file_sha256=file_hash(weights),
            max_seconds=30,
            costs=[0.05, 0.05],
            correlations=[0.9, 0.0],
            defer_bounds=[0.2, 0.8],
            sigmas=[0.03, 0.15, 0.3, 0.6],
        ),
        data=dict(generator="gate_shift_examples", pairs=384, seeds=[1601, 2601]),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_source_choice.json"
    started = perf_counter()
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(model):
                raise ValueError("Evidence source cache mismatch")
        else:
            result = evaluate_source_choice(model)
            if state_hash(model) != before:
                raise RuntimeError("Frozen source gate changed")
            if perf_counter() - started > 30:
                raise TimeoutError("Evidence source budget exhausted")
            result["model_sha256"] = before
            atomic_json(path, result)
            for name, scores in result["summary"].items():
                run.log(
                    dict(
                        step=0,
                        split="source_choice",
                        condition=name,
                        accuracy=scores["accuracy"],
                        utility=scores["utility"],
                    )
                )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_evidence_sources(weights, output, resume=False):
    """Fixed same-source versus alternate-source acquisition; no fitting."""
    from pathwm.models.entity_relations import RelationWriteGate
    from pathwm.evaluation.entity_gate import score_evidence_sources

    weights = Path(weights).resolve()
    model = RelationWriteGate().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True)["model"]
    )
    before = state_hash(model)
    run = Run(
        output,
        settings=dict(
            seed=1501,
            innovation_seed=1502,
            bootstrap_seed=1503,
            pairs=128,
            purpose="diagnostic",
            entity_evidence_sources=True,
            entity_gate_weights=str(weights),
            gate_file_sha256=file_hash(weights),
            max_seconds=30,
            costs=[0.02, 0.05],
            correlations=[0.9, 0.0],
            defer_bounds=[0.2, 0.8],
            sigmas=[0.03, 0.15, 0.3, 0.6],
        ),
        data=dict(generator="gate_shift_examples", pairs=128, seeds=[1501, 1502, 1503]),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_evidence_sources.json"
    started = perf_counter()
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(model):
                raise ValueError("Evidence source cache mismatch")
        else:
            result = score_evidence_sources(model)
            if state_hash(model) != before:
                raise RuntimeError("Frozen source gate changed")
            if perf_counter() - started > 30:
                raise TimeoutError("Evidence source budget exhausted")
            result["model_sha256"] = before
            atomic_json(path, result)
            for name, source in result["sources"].items():
                for noise, c in source["cohorts"].items():
                    v = c["strategies"]["selective"]
                    run.log(
                        dict(
                            step=0,
                            split="frozen_evidence_sources",
                            condition=name,
                            noise=noise,
                            accuracy=v["accuracy"],
                            utility=v["utility"],
                        )
                    )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_gate_shift(
    weights, output, resume=False, *, reobserve=False, correlation=None
):
    """Frozen context-noise sensitivity; no optimization."""
    from pathwm.models.entity_relations import RelationWriteGate
    from pathwm.evaluation.entity_gate import (
        gate_shift_examples,
        score_gate_shift,
        gate_reobserve_examples,
        score_gate_reobserve,
    )

    if correlation is not None and not reobserve:
        raise ValueError("Correlation requires reobservation")
    weights = Path(weights).resolve()
    model = RelationWriteGate().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True)["model"]
    )
    before = state_hash(model)
    data = (
        gate_reobserve_examples(correlation=correlation)
        if reobserve
        else gate_shift_examples()
    )
    run = Run(
        output,
        settings=dict(
            seed=(1301 if correlation is None else 1401) if reobserve else 801,
            entity_gate_correlation=correlation,
            entity_gate_reobserve=reobserve,
            reread_seed=(1302 if correlation is None else 1402) if reobserve else None,
            defer_bounds=[0.2, 0.8] if reobserve else None,
            observation_cost=0.02 if reobserve else None,
            purpose="diagnostic",
            entity_gate_shift=True,
            entity_gate_weights=str(weights),
            gate_file_sha256=file_hash(weights),
            max_seconds=30,
            sigmas=[0.03, 0.15, 0.30, 0.60],
            thresholds=[0.4, 0.5, 0.6],
            pairs=128,
        ),
        data=dict(contexts=digest({k: v.tolist() for k, v in data.items()})),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_gate_shift.json"
    started = perf_counter()
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(model):
                raise ValueError("Gate shift cache mismatch")
        else:
            result = (
                score_gate_reobserve(model, data)
                if reobserve
                else score_gate_shift(model, data)
            )
            if state_hash(model) != before:
                raise RuntimeError("Frozen gate changed")
            if perf_counter() - started > 30:
                raise TimeoutError("Gate shift budget exhausted")
            if correlation is not None:
                result["correlation"] = correlation
                result["noise_diagnostics"] = dict(
                    first_norm_mean=data["noise"].norm(dim=-1).mean().item(),
                    second_norm_mean=data["second_noise"].norm(dim=-1).mean().item(),
                    empirical_raw_correlation=torch.corrcoef(
                        torch.stack(
                            [data["noise"].flatten(), data["second_noise"].flatten()]
                        )
                    )[0, 1].item(),
                )
                if correlation == 1:
                    result["passed"] = all(
                        c["strategies"]["first"]["probability"]
                        == c["strategies"]["selective"]["probability"]
                        or [v > 0.5 for v in c["strategies"]["first"]["probability"]]
                        == [
                            v > 0.5 for v in c["strategies"]["selective"]["probability"]
                        ]
                        for c in result["cohorts"].values()
                    )
            result["model_sha256"] = before
            atomic_json(path, result)
            for name, c in result["cohorts"].items():
                run.log(
                    dict(
                        step=0,
                        split="frozen_gate_shift",
                        condition=name,
                        accuracy=c["thresholds"]["0.5"]["accuracy"],
                        brier=c["brier"],
                    )
                )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def gate_replication_seeds(replicate):
    if type(replicate) is not int or replicate < 0:
        raise ValueError("Replicate must be a nonnegative integer")
    return 71 + replicate, 100 * replicate


def train_entity_gate(
    weights,
    cell_weights,
    key_weights,
    output,
    resume=False,
    *,
    gate_weights=None,
    augmented=False,
    replicate=0,
    retention=0.0,
):
    """Fit only context relevance through frozen source-selection loss."""
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityInteractionCell
    from pathwm.models.entity_relations import RelationKey, RelationWriteGate
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_gate import gate_examples, gate_logits, evaluate_gate

    from pathwm.evaluation.entity_gate import (
        augmented_gate_examples,
        gate_shift_examples,
        score_gate_shift,
        gate_retention_loss,
    )

    if augmented and gate_weights is None:
        raise ValueError("Augmentation requires a gate donor")
    if not 0 <= retention <= 100 or (
        retention and (gate_weights is None or not augmented)
    ):
        raise ValueError(
            "Retention requires noisy continuation and a finite weight in [0,100]"
        )
    continuation = gate_weights is not None
    replication_seed, offset = gate_replication_seeds(replicate)
    if replicate and not continuation:
        raise ValueError("Replication requires a gate donor")
    seed_everything(replication_seed if continuation else 61)
    weights, cell_weights = Path(weights).resolve(), Path(cell_weights).resolve()
    matcher = EntityMatchReader().eval().requires_grad_(False)
    donor = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    matcher.load_state_dict(
        {
            k.removeprefix("agent."): v
            for k, v in donor.items()
            if k.startswith("agent.")
        }
    )
    cell = EntityInteractionCell().eval().requires_grad_(False)
    cell.load_state_dict(
        torch.load(cell_weights, map_location="cpu", weights_only=True)["model"]
    )
    if bool(cell._interaction_blind):
        raise ValueError("Relations require a source-aware interaction")
    key_weights = Path(key_weights).resolve()
    key = RelationKey().eval().requires_grad_(False)
    key.load_state_dict(
        torch.load(key_weights, map_location="cpu", weights_only=True)["model"]
    )
    gate = RelationWriteGate()
    if continuation:
        gate_weights = Path(gate_weights).resolve()
        gate.load_state_dict(
            torch.load(gate_weights, map_location="cpu", weights_only=True)["model"]
        )
    initial_gate_sha256 = state_hash(gate)
    shift_data = gate_shift_examples(921 + offset) if continuation else None
    shift_before = score_gate_shift(gate, shift_data) if continuation else None
    before = (state_hash(matcher), state_hash(cell), state_hash(key))
    families = {
        name: growth_inputs(seed + (300 + offset if continuation else 0), count)
        for name, seed, count in [
            ("train", 601, 32),
            ("development", 602, 8),
            ("evaluation", 603, 8),
        ]
    }
    train = (
        augmented_gate_examples(families["train"], 911 + offset, augmented)
        if continuation
        else gate_examples(families["train"], 701)
    )
    dev = gate_examples(families["development"], 912 + offset if continuation else 702)
    evaluation = gate_examples(
        families["evaluation"], 913 + offset if continuation else 703
    )
    clean = (
        augmented_gate_examples(families["train"], 911 + offset, False)
        if continuation
        else None
    )
    with torch.no_grad():
        teacher = (
            gate(clean["active"], clean["cue"]).detach().clone()
            if continuation
            else None
        )
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.01, weight_decay=0.01)
    run = Run(
        output,
        settings=dict(
            seed=replication_seed if continuation else 61,
            entity_gate_replicate=replicate,
            entity_gate_retain=retention,
            entity_gate_weights=str(gate_weights) if continuation else None,
            entity_gate_augment=augmented,
            initial_gate_sha256=initial_gate_sha256,
            gate_file_sha256=file_hash(gate_weights) if continuation else None,
            purpose="diagnostic",
            entity_gate=True,
            entity_relation_key=str(key_weights),
            key_file_sha256=file_hash(key_weights),
            context_seeds=[911 + offset, 912 + offset, 913 + offset, 921 + offset]
            if continuation
            else [701, 702, 703],
            entity_state_weights=str(weights),
            entity_temporal_cell=str(cell_weights),
            matcher_sha256=file_hash(weights),
            cell_sha256=file_hash(cell_weights),
            steps=256,
            batch_size=32,
            learning_rate=0.01,
            max_seconds=450,
        ),
        data={name: digest(f) for name, f in families.items()},
        recipe=__file__,
        model=gate,
        optimizer=optimizer,
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_gate.json"
    deadline = perf_counter() + 450
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["gate_sha256"] != state_hash(gate):
                raise ValueError("Relation cache mismatch")
        else:
            for step in range(256):
                if perf_counter() > deadline:
                    raise TimeoutError("Relation training budget exhausted")
                ix = torch.randint(len(train["labels"]), (32,), generator=run.sampler)
                logits, _ = gate_logits(
                    gate, key, matcher, {k: v[ix] for k, v in train.items()}
                )
                source_loss = F.cross_entropy(logits, train["labels"][ix])
                retention_loss = (
                    gate_retention_loss(
                        gate(clean["active"][ix], clean["cue"][ix]), teacher[ix]
                    )
                    if continuation
                    else source_loss.new_zeros(())
                )
                loss = source_loss + retention * retention_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run.step = step + 1
                run.log(
                    dict(
                        step=run.step,
                        split="train",
                        loss=loss.item(),
                        source_loss=source_loss.item(),
                        retention_loss=retention_loss.item(),
                    )
                )
            with torch.inference_mode():
                logits, _ = gate_logits(gate, key, matcher, dev)
                dev_scores = dict(
                    accuracy=(logits.argmax(-1) == dev["labels"]).float().mean().item(),
                    nll=F.cross_entropy(logits, dev["labels"]).item(),
                    examples=len(dev["labels"]),
                )
            result = evaluate_gate(
                matcher, cell, key, gate, families["evaluation"], evaluation, deadline
            )
            if (state_hash(matcher), state_hash(cell), state_hash(key)) != before:
                raise RuntimeError("Frozen relation consumers changed")
            result.update(
                retention_weight=retention,
                development=dev_scores,
                gate_sha256=state_hash(gate),
                evaluation={k: v.tolist() for k, v in evaluation.items()},
                frozen_sha256=before,
                families=families,
            )
            result["passed"] &= (
                dev_scores["accuracy"] >= 0.95 and dev_scores["nll"] <= 0.15
            )
            if continuation:
                after = score_gate_shift(gate, shift_data)
                high = after["cohorts"]["0.6"]["thresholds"]["0.5"]
                base = shift_before["cohorts"]["0.6"]["thresholds"]["0.5"]
                low_ok = all(
                    min(
                        after["cohorts"][n]["thresholds"]["0.5"][k]
                        for k in ("positive_recall", "negative_recall")
                    )
                    >= 0.95
                    for n in ("0.03", "0.15")
                )
                adaptation_passed = (
                    low_ok
                    and high["accuracy"] >= base["accuracy"] + 0.02
                    and high["negative_recall"] >= base["negative_recall"] - 0.05
                )
                result.update(
                    shift_before=shift_before,
                    shift_after=after,
                    adaptation_passed=adaptation_passed,
                    augmented=augmented,
                    initial_gate_sha256=initial_gate_sha256,
                )
                result["passed"] &= adaptation_passed
            atomic_json(path, result)
            run.log(
                dict(
                    step=run.step,
                    split="validation",
                    loss=dev_scores["nll"],
                    accuracy=dev_scores["accuracy"],
                )
            )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def train_entity_relations(weights, cell_weights, output, resume=False):
    """Fit only relation addressing; persistence and relation type are explicit."""
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityInteractionCell
    from pathwm.models.entity_relations import RelationKey
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_relations import relation_examples, evaluate_relations

    seed_everything(51)
    weights, cell_weights = Path(weights).resolve(), Path(cell_weights).resolve()
    matcher = EntityMatchReader().eval().requires_grad_(False)
    donor = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    matcher.load_state_dict(
        {
            k.removeprefix("agent."): v
            for k, v in donor.items()
            if k.startswith("agent.")
        }
    )
    cell = EntityInteractionCell().eval().requires_grad_(False)
    cell.load_state_dict(
        torch.load(cell_weights, map_location="cpu", weights_only=True)["model"]
    )
    if bool(cell._interaction_blind):
        raise ValueError("Relations require a source-aware interaction")
    key = RelationKey()
    before = (state_hash(matcher), state_hash(cell))
    families = {
        name: growth_inputs(seed, count)
        for name, seed, count in [
            ("train", 501, 32),
            ("development", 502, 8),
            ("evaluation", 503, 8),
        ]
    }
    train = relation_examples(families["train"])
    dev = relation_examples(families["development"])
    optimizer = torch.optim.AdamW(key.parameters(), lr=0.01, weight_decay=0.01)
    run = Run(
        output,
        settings=dict(
            seed=51,
            purpose="diagnostic",
            entity_relations=True,
            entity_state_weights=str(weights),
            entity_temporal_cell=str(cell_weights),
            matcher_sha256=file_hash(weights),
            cell_sha256=file_hash(cell_weights),
            steps=256,
            batch_size=32,
            learning_rate=0.01,
            max_seconds=450,
        ),
        data={name: digest(f) for name, f in families.items()},
        recipe=__file__,
        model=key,
        optimizer=optimizer,
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_relations.json"
    deadline = perf_counter() + 450
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["key_sha256"] != state_hash(key):
                raise ValueError("Relation cache mismatch")
        else:
            for step in range(256):
                if perf_counter() > deadline:
                    raise TimeoutError("Relation training budget exhausted")
                ix = torch.randint(len(train["labels"]), (32,), generator=run.sampler)
                logits = matcher.match(key(train["cues"][ix]), train["candidates"][ix])
                loss = F.cross_entropy(logits, train["labels"][ix])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run.step = step + 1
                run.log(dict(step=run.step, split="train", loss=loss.item()))
            with torch.inference_mode():
                logits = matcher.match(key(dev["cues"]), dev["candidates"])
                dev_scores = dict(
                    accuracy=(logits.argmax(-1) == dev["labels"]).float().mean().item(),
                    nll=F.cross_entropy(logits, dev["labels"]).item(),
                    examples=len(dev["labels"]),
                )
            result = evaluate_relations(
                matcher, cell, key, families["evaluation"], deadline
            )
            if (state_hash(matcher), state_hash(cell)) != before:
                raise RuntimeError("Frozen relation consumers changed")
            result.update(
                development=dev_scores,
                key_sha256=state_hash(key),
                frozen_sha256=before,
                families=families,
            )
            result["passed"] &= (
                dev_scores["accuracy"] >= 0.95 and dev_scores["nll"] <= 0.15
            )
            atomic_json(path, result)
            run.log(
                dict(
                    step=run.step,
                    split="validation",
                    loss=dev_scores["nll"],
                    accuracy=dev_scores["accuracy"],
                )
            )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_source(weights, cell_weights, output, resume=False):
    """One frozen retrieval integration screen; no parameter updates."""
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityInteractionCell
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_source import evaluate_sources

    seed_everything(401)
    weights, cell_weights = Path(weights).resolve(), Path(cell_weights).resolve()
    matcher = EntityMatchReader()
    donor = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    matcher.load_state_dict(
        {
            k.removeprefix("agent."): v
            for k, v in donor.items()
            if k.startswith("agent.")
        }
    )
    cell = EntityInteractionCell()
    cell.load_state_dict(
        torch.load(cell_weights, map_location="cpu", weights_only=True)["model"]
    )
    if bool(cell._interaction_blind):
        raise ValueError("Source retrieval requires a source-aware interaction donor")
    models = (
        nn.ModuleDict(dict(matcher=matcher, cell=cell)).eval().requires_grad_(False)
    )
    before = state_hash(models)
    families = growth_inputs(401, 8)
    run = Run(
        output,
        settings=dict(
            seed=401,
            purpose="diagnostic",
            entity_source=True,
            entity_state_weights=str(weights),
            entity_temporal_cell=str(cell_weights),
            matcher_sha256=file_hash(weights),
            cell_sha256=file_hash(cell_weights),
            max_seconds=180,
        ),
        data=dict(families=digest(families)),
        recipe=__file__,
        model=models,
        optimizer=torch.optim.AdamW(models.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_source.json"
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(models):
                raise ValueError("Source cache mismatch")
        else:
            result = evaluate_sources(matcher, cell, families)
            if state_hash(models) != before:
                raise RuntimeError("Frozen retrieval models changed")
            result["model_sha256"] = before
            result["families"] = families
            atomic_json(path, result)
            for name, c in result["cohorts"].items():
                run.log(
                    dict(
                        step=0,
                        split="frozen_source",
                        condition=name,
                        accuracy=c["accuracy"],
                        nll=c["nll"],
                        source_accuracy=c["source_accuracy"],
                    )
                )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def train_entity_interaction(weights, cell_weights, output, resume=False, blind=False):
    """One bounded interaction fit with frozen recognition and ordinary state dynamics."""
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityInteractionCell
    from pathwm.data.entity_interaction import interaction_episodes
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_state import state_metrics, state_runtime

    seed_everything(41)
    weights, cell_weights = Path(weights).resolve(), Path(cell_weights).resolve()
    matcher = EntityMatchReader().eval().requires_grad_(False)
    donor = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    matcher.load_state_dict(
        {
            k.removeprefix("agent."): v
            for k, v in donor.items()
            if k.startswith("agent.")
        }
    )
    cell = EntityInteractionCell(blind=blind)
    base = torch.load(cell_weights, map_location="cpu", weights_only=True)["model"]
    expected = {
        k
        for k in cell.state_dict()
        if not k.startswith("interaction.") and k != "_interaction_blind"
    }
    if set(base) != expected or not bool(base["_preserve_no_information"]):
        raise ValueError("Interaction requires the explicit idle-preserving base cell")
    cell.load_state_dict({**cell.state_dict(), **base})

    def frozen():
        return (state_hash(matcher), state_hash(cell.cell), state_hash(cell.head))

    before = frozen()
    training = interaction_episodes(matcher, growth_inputs(301, 32), 311)
    development = interaction_episodes(matcher, growth_inputs(302, 16), 312)
    families = growth_inputs(303, 16)
    populations = {
        name: interaction_episodes(matcher, families, 313, name)
        for name in ("reference", "swapped", "idle", "composition")
    }
    optimizer = torch.optim.AdamW(
        cell.interaction.parameters(), lr=0.003, weight_decay=0.01
    )
    settings = dict(
        seed=41,
        purpose="diagnostic",
        entity_interaction=True,
        entity_interaction_blind=blind,
        entity_state_weights=str(weights),
        entity_temporal_cell=str(cell_weights),
        donor_sha256=file_hash(weights),
        state_sha256=file_hash(cell_weights),
        steps=256,
        batch_size=32,
        max_seconds=450,
        learning_rate=0.003,
        width=16,
    )
    run = Run(
        output,
        settings=settings,
        data={
            k: digest(v["manifest"])
            for k, v in dict(
                train=training, development=development, **populations
            ).items()
        },
        recipe=__file__,
        model=cell,
        optimizer=optimizer,
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_interaction.json"
    deadline = perf_counter() + 450
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["cell_sha256"] != state_hash(cell):
                raise ValueError("Interaction cache mismatch")
        else:
            for step in range(256):
                if perf_counter() > deadline:
                    raise TimeoutError("Interaction training budget exhausted")
                ix = torch.randint(len(training["slots"]), (32,), generator=run.sampler)
                logits, _ = cell(
                    training["observations"][ix],
                    training["slots"][ix],
                    training["sources"][ix],
                )
                loss = F.cross_entropy(
                    logits.flatten(0, 1), training["targets"][ix].flatten()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run.step = step + 1
                run.log(dict(step=run.step, split="train", loss=loss.item()))
            train_scores, _, _ = state_metrics(cell, training)
            cohorts = {}
            for name, data in dict(development=development, **populations).items():
                scores, logits, _ = state_metrics(cell, data)
                runtime = state_runtime(cell, matcher, data, deadline=deadline)
                passed = (
                    scores["pair_accuracy"] >= 0.95
                    and scores["nll"] <= 0.15
                    and runtime["pair_accuracy"] >= 0.95
                    and runtime["transactions"]
                    and runtime["latent_agreement"]
                )
                cohorts[name] = dict(
                    scores=scores,
                    runtime=runtime,
                    passed=passed,
                    manifest=data["manifest"],
                    logits=logits.tolist(),
                )
                run.log(
                    dict(
                        step=run.step,
                        split="validation"
                        if name == "development"
                        else "frozen_interaction",
                        condition=name,
                        loss=scores["nll"],
                        pair_accuracy=scores["pair_accuracy"],
                    )
                )
            if frozen() != before:
                raise RuntimeError("Frozen recognition or state dynamics changed")
            result = dict(
                train=train_scores,
                cohorts=cohorts,
                blind=blind,
                cell_sha256=state_hash(cell),
                frozen_sha256=before,
                passed=all(c["passed"] for c in cohorts.values()),
            )
            atomic_json(path, result)
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def train_entity_state(weights, output, resume=False, varied=False, preserve=False):
    """One fixed learned-state diagnostic with a frozen recognition component."""
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityStateCell
    from pathwm.data.entity_state import state_episodes, mixed_state_episodes
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_state import state_metrics, state_runtime

    seed_everything(31)
    weights = Path(weights).resolve()
    donor = torch.load(weights, map_location="cpu", weights_only=True)["model"]
    donor = {
        k.removeprefix("agent."): v for k, v in donor.items() if k.startswith("agent.")
    }
    matcher = EntityMatchReader()
    matcher.load_state_dict(donor)
    matcher.eval().requires_grad_(False)
    matcher_hash = state_hash(matcher)
    if varied:
        training = mixed_state_episodes(
            matcher, growth_inputs(seed=201, count=32), seed=211
        )
        development = mixed_state_episodes(
            matcher, growth_inputs(seed=202, count=16), seed=212
        )
    else:
        training = state_episodes(matcher, growth_inputs(seed=101, count=32))
        development = state_episodes(matcher, growth_inputs(seed=102, count=16))
    cell = EntityStateCell(preserve_no_information=preserve)
    optimizer = torch.optim.AdamW(cell.parameters(), lr=0.003, weight_decay=0.01)
    settings = dict(
        seed=31,
        purpose="diagnostic",
        entity_state_weights=str(weights),
        entity_state_varied=varied,
        entity_state_preserve=preserve,
        donor_sha256=file_hash(weights),
        steps=256,
        batch_size=32,
        max_seconds=450,
        learning_rate=0.003,
        width=16,
    )
    run = Run(
        output,
        settings=settings,
        data={
            k: digest(v["manifest"])
            for k, v in [("train", training), ("development", development)]
        },
        recipe=__file__,
        model=cell,
        optimizer=optimizer,
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_state.json"
    deadline = perf_counter() + 450
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["cell_sha256"] != state_hash(cell):
                raise ValueError("State cache mismatch")
        else:
            for step in range(256):
                if perf_counter() > deadline:
                    raise TimeoutError("State training budget exhausted")
                indices = torch.randint(
                    len(training["slots"]), (32,), generator=run.sampler
                )
                logits, _ = cell(
                    training["observations"][indices], training["slots"][indices]
                )
                loss = F.cross_entropy(
                    logits.flatten(0, 1), training["targets"][indices].flatten()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run.step = step + 1
                run.log(dict(step=run.step, split="train", loss=loss.item()))
            train_scores, _, _ = state_metrics(cell, training)
            dev_scores, logits, _ = state_metrics(cell, development)
            runtime = state_runtime(cell, matcher, development)
            if state_hash(matcher) != matcher_hash:
                raise RuntimeError("Frozen recognizer changed")
            result = dict(
                preserve_no_information=preserve,
                train=train_scores,
                development=dev_scores,
                runtime=runtime,
                cell_sha256=state_hash(cell),
                matcher_sha256=matcher_hash,
                manifest=development["manifest"],
                logits=logits.tolist(),
            )
            result["passed"] = (
                dev_scores["pair_accuracy"] >= 0.95
                and dev_scores["nll"] <= 0.15
                and runtime["pair_accuracy"] >= 0.95
                and runtime["transactions"]
                and runtime["latent_agreement"]
            )
            atomic_json(path, result)
            run.log(
                dict(
                    step=run.step,
                    split="validation",
                    loss=dev_scores["nll"],
                    pair_accuracy=dev_scores["pair_accuracy"],
                )
            )
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def evaluate_entity_temporal(
    matcher_weights,
    cell_weights,
    output,
    resume=False,
    seed=111,
    varied=False,
    idle=False,
):
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityStateCell
    from pathwm.data.entity_temporal import temporal_episodes
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_state import state_metrics, state_runtime

    seed_everything(seed)
    matcher_weights, cell_weights = (
        Path(matcher_weights).resolve(),
        Path(cell_weights).resolve(),
    )
    donor = torch.load(matcher_weights, map_location="cpu", weights_only=True)["model"]
    matcher = EntityMatchReader()
    matcher.load_state_dict(
        {
            k.removeprefix("agent."): v
            for k, v in donor.items()
            if k.startswith("agent.")
        }
    )
    cell_state = torch.load(cell_weights, map_location="cpu", weights_only=True)[
        "model"
    ]
    cell = EntityStateCell(
        preserve_no_information="_preserve_no_information" in cell_state
    )
    cell.load_state_dict(cell_state)
    models = (
        nn.ModuleDict(dict(matcher=matcher, cell=cell)).eval().requires_grad_(False)
    )
    before = state_hash(models)
    deadline = perf_counter() + (240 if idle else 120)
    families = growth_inputs(seed, 16)
    populations = temporal_episodes(
        matcher, families, idle_lengths=(3, 31) if idle else ()
    )
    if varied:
        from pathwm.data.entity_state import mixed_state_episodes

        populations["unseen_composition"] = mixed_state_episodes(
            matcher, families, seed=221, middle_steps=16
        )
    settings = dict(
        seed=seed,
        entity_temporal_seed=seed,
        entity_temporal_varied=varied,
        entity_temporal_idle=idle,
        purpose="diagnostic",
        entity_state_weights=str(matcher_weights),
        entity_temporal_cell=str(cell_weights),
        matcher_sha256=file_hash(matcher_weights),
        cell_sha256=file_hash(cell_weights),
        max_seconds=240 if idle else 120,
    )
    run = Run(
        output,
        settings=settings,
        data={k: digest(v["manifest"]) for k, v in populations.items()},
        recipe=__file__,
        model=models,
        optimizer=torch.optim.AdamW(models.parameters(), lr=0),
        device="cpu",
        resume=resume,
    )
    path = run.path / "entity_temporal.json"
    try:
        if resume:
            result = json.loads(path.read_text())
            if result["model_sha256"] != state_hash(models):
                raise ValueError("Temporal cache mismatch")
        else:
            cohorts = {}
            for name, data in populations.items():
                scores, logits, _ = state_metrics(cell, data)
                runtime = state_runtime(cell, matcher, data, deadline=deadline)
                passed = (
                    scores["pair_accuracy"] >= 0.95
                    and scores["nll"] <= 0.15
                    and runtime["pair_accuracy"] >= 0.95
                    and runtime["transactions"]
                    and runtime["latent_agreement"]
                )
                cohorts[name] = dict(
                    scores=scores,
                    runtime=runtime,
                    passed=passed,
                    manifest=data["manifest"],
                    logits=logits.tolist(),
                )
                run.log(dict(step=0, split="frozen_temporal", condition=name, **scores))
            if state_hash(models) != before:
                raise RuntimeError("Frozen models changed")
            result = dict(
                preserve_no_information=bool(
                    getattr(cell, "_preserve_no_information", False)
                ),
                cohorts=cohorts,
                model_sha256=before,
                passed=all(c["passed"] for c in cohorts.values()),
            )
            atomic_json(path, result)
            run.save()
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def build_visual_memory():
    from pathwm.models.visual_memory import VisualMemoryReader

    return VisualMemoryReader(
        build_model(
            width=16,
            image_size=32,
            levels=1,
            state_model="belief",
            memory_recent=2,
            memory_block=2,
            memory_blocks=1,
        )
    )


@torch.no_grad()
def direct_visual_weights(
    output,
    *,
    fit_pairs=32,
    development_pairs=16,
    test_pairs=32,
    device="cpu",
    max_seconds=600,
    resume=False,
    timing=False,
    seed_offset=0,
):
    """Construct and test weight files without gradients or optimizer updates."""
    import shutil
    from pathwm.data.visual_memory import VisualMemoryEpisodes
    from pathwm.models.visual_memory import construct_weights, fit_centroid_head
    from pathwm.evaluation.visual_memory import visual_features, visual_metrics

    if (
        min(fit_pairs, development_pairs, test_pairs) < 1
        or not np.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Positive populations and finite time cap required")
    device = str(device)
    if torch.device(device).type == "cuda":
        if torch.device(device).index is None:
            device = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.set_per_process_memory_fraction(
            2**30 / torch.cuda.get_device_properties(device).total_memory, device
        )
        torch.cuda.reset_peak_memory_stats(device)
    seed_everything(3001)
    fit, development, test = [
        VisualMemoryEpisodes(n, seed=s + seed_offset)
        for n, s in [
            (fit_pairs, 3101),
            (development_pairs, 3201),
            (test_pairs, 5301 if timing else 3301),
        ]
    ]
    for key in ("scene_sha256", "final_sha256"):
        groups = [set(d.identity[key]) for d in (fit, development, test)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Visual-memory split overlap")
    model = build_visual_memory().to(device).eval()
    # Unused, zero-lr optimizer exists only for the existing checkpoint schema.
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    settings = dict(
        dataset="direct-weights",
        direct_weight_timing=timing,
        seed_offset=seed_offset,
        seed=3001,
        fit_pairs=fit_pairs,
        development_pairs=development_pairs,
        test_pairs=test_pairs,
        device=device,
        max_seconds=max_seconds,
        purpose="diagnostic",
        precision="float32",
        method="direct_hand_construction_then_label_fitted_centroid_head",
        optimizer_updates=0,
        backward_calls=0,
        gpu_allocator_cap_bytes=2**30,
    )
    run = Run(
        output,
        settings=settings,
        data={
            k: d.identity
            for k, d in zip(("fit", "development", "test"), (fit, development, test))
        },
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    path = run.path / "visual_memory.json"
    started = perf_counter()
    deadline = started + max_seconds

    def scores(data, seed, erased=False):
        features = visual_features(
            model, data, device=device, seed=seed, erased=erased, deadline=deadline
        )
        return visual_metrics(model.head(features.to(device)), data)

    def save_weights(name, method):
        target = run.path / name
        torch.save(
            dict(
                schema="pathwm-visual-weights-v1",
                architecture="build_visual_memory(width16, RGB32, four frames)",
                method=method,
                optimizer_updates=0,
                model={
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
            ),
            target,
        )
        expected = state_hash(model)
        loaded = torch.load(target, map_location=device, weights_only=True)
        model.load_state_dict(loaded["model"], strict=True)
        if state_hash(model) != expected:
            raise RuntimeError("Serialized weights do not reproduce state")
        return dict(
            path=name, sha256=file_hash(target), model_sha256=expected, method=method
        )

    try:
        if resume and path.exists():
            cached = json.loads(path.read_text())
            if (
                state_hash(model) != cached["model_sha256"]
                or file_hash(run.path / "weights.pt") != cached["weights_sha256"]
            ):
                raise ValueError("Direct-weight result cache mismatch")
        else:
            baseline_file = save_weights(
                "ordinary_initialization.pt", "ordinary_seed3001_initialization"
            )
            baseline_development = scores(development, 3501)
            candidates = []
            construct_weights(model, timing=timing)
            constructed = save_weights(
                "constructed.pt",
                "handwritten_numeric_assignment"
                + ("_timing_revision" if timing else ""),
            )
            candidates.append(
                dict(
                    name="constructed",
                    file=constructed,
                    development=scores(development, 3501),
                )
            )
            fit_features = visual_features(
                model, fit, device=device, seed=3601, deadline=deadline
            )
            fit_centroid_head(model, fit_features, torch.from_numpy(fit.labels))
            adjusted = save_weights(
                "adjusted.pt",
                "label_fitted_centroid_head_on_constructed_backbone"
                + ("_timing_revision" if timing else ""),
            )
            candidates.append(
                dict(
                    name="adjusted",
                    file=adjusted,
                    development=scores(development, 3501),
                )
            )
            selected = max(
                candidates,
                key=lambda c: (c["development"]["accuracy"], -c["development"]["nll"]),
            )
            # Persist selection BEFORE any final-test forward pass.
            atomic_json(
                run.path / "selection.json",
                dict(
                    selected=selected["name"],
                    rule="development accuracy, then NLL, stable candidate order",
                    candidates=candidates,
                ),
            )
            model.load_state_dict(
                torch.load(
                    run.path / selected["file"]["path"],
                    map_location=device,
                    weights_only=True,
                )["model"]
            )
            shutil.copyfile(
                run.path / selected["file"]["path"], run.path / "weights.pt"
            )
            evaluation = scores(test, 3401)
            erased = scores(test, 3401, erased=True)
            if not np.array_equal(
                np.asarray(erased["logits"])[::2], np.asarray(erased["logits"])[1::2]
            ):
                raise RuntimeError(
                    "Identical paired erased histories produced different logits"
                )
            selected_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            model.load_state_dict(
                torch.load(
                    run.path / "ordinary_initialization.pt",
                    map_location=device,
                    weights_only=True,
                )["model"]
            )
            baseline = scores(test, 3401)
            model.load_state_dict(selected_state)
            peak = (
                dict(
                    allocated_bytes=torch.cuda.max_memory_allocated(device),
                    reserved_bytes=torch.cuda.max_memory_reserved(device),
                )
                if torch.device(device).type == "cuda"
                else None
            )
            if peak is not None and peak["reserved_bytes"] > 2**30:
                raise RuntimeError("GPU allocator peak exceeds pilot budget")
            gates = dict(
                accuracy=evaluation["accuracy"] >= 0.9,
                pair_both=evaluation["pair_both"] >= 0.8,
                reversal=evaluation["reversal"]["accuracy"] is not None
                and evaluation["reversal"]["accuracy"] >= 0.8,
                history_advantage=evaluation["accuracy"] - erased["accuracy"] >= 0.3,
            )
            result = dict(
                selected=selected["name"],
                candidates=candidates,
                evaluation=evaluation,
                erased=erased,
                baseline=baseline,
                baseline_development=baseline_development,
                baseline_file=baseline_file,
                gates=gates,
                passed=all(gates.values()),
                model_sha256=state_hash(model),
                weights_sha256=file_hash(run.path / "weights.pt"),
                optimizer_updates=0,
                backward_calls=0,
                inference_gpu_peak=peak,
                parameter_count=sum(p.numel() for p in model.parameters()),
                active_seconds=perf_counter() - started,
                limits="Task-specific synthetic four-frame recall; no webcam transfer, general entity learning or directly generated pretrained model demonstrated.",
            )
            if result["active_seconds"] > max_seconds:
                raise TimeoutError("Direct-weight evaluation budget exhausted")
            np.savez_compressed(
                run.path / "visual_examples.npz",
                images=test.images[:8],
                labels=test.labels[:8],
            )
            atomic_json(
                run.path / "visual_data.json",
                {
                    k: dict(identity=d.identity, records=d.records)
                    for k, d in zip(
                        ("fit", "development", "test"), (fit, development, test)
                    )
                },
            )
            for i, candidate in enumerate(candidates):
                run.log(
                    dict(
                        step=i,
                        split="validation",
                        loss=candidate["development"]["nll"],
                        accuracy=candidate["development"]["accuracy"],
                    )
                )
            run.save()
            atomic_json(path, result)
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        render_report(run.path)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise
    return run.path


def capability_models(overrides=None):
    """Explicit roles and strict loads; these checkpoints are NOT one trained agent."""
    from pathwm.models.key_box import KeyBoxReader
    from pathwm.models.entity_state import EntityStateCell, EntityInteractionCell
    from pathwm.models.entity_relations import RelationKey, RelationWriteGate
    from pathwm.models.encoders import CNNEncoder
    from pathwm.models.decoders import ReconstructionDecoder, PushTPoseHead
    from pathwm.models.perception import Perception
    from pathwm.models.temporal import MemoryUpdater, Predictor, WorldModel

    paths = dict(
        visual="runs/direct_weights_v1/timing/weights.pt",
        synthetic="runs/belief_v1/synthetic/last.pt",
        instructions="runs/belief_v1/instructions/last.pt",
        pusht="runs/belief_v1/pusht/last.pt",
        facts_direct="runs/fact_grounding_v1/reference/last.pt",
        facts_event="runs/event_fact_v1/lr_control/last.pt",
        recall="runs/recall_v1/development/last.pt",
        key_box="runs/key_box_v1/replica/last.pt",
        interaction="runs/entity_interaction_v1/frozen_interaction.pt",
        relation="runs/entity_relations_v1/frozen_key.pt",
        gate="runs/entity_gate_v1/frozen_gate.pt",
        perception="runs/examples/perception/last.pt",
        dynamics="runs/examples/dynamics/last.pt",
    )
    if set(overrides or {}) - set(paths):
        raise ValueError("Unknown checkpoint role")
    paths.update(overrides or {})
    models, settings, receipts = nn.ModuleDict(), {}, {}
    for name, filename in paths.items():
        path = Path(filename).resolve()
        sha = file_hash(path)
        prefix = ""
        if name in (
            "synthetic",
            "instructions",
            "pusht",
            "facts_direct",
            "facts_event",
            "recall",
        ):
            settings[name] = json.loads((path.parent / "run.json").read_text())[
                "identity"
            ]["settings"]
            s = settings[name]
            model = build_model(
                s["width"],
                s["image_size"],
                s["audio_samples"],
                state_model=s["state_model"],
                memory_recent=s["memory_recent"],
                memory_block=s["memory_block"],
                memory_blocks=s["memory_blocks"],
                facts=name.startswith("facts_"),
                fact_reader=s.get("fact_reader", "direct"),
                recall=name == "recall",
            )
            # Existing recall recipe deploys its development-selected target copy.
            prefix = "target." if name == "recall" else "agent."
        elif name == "visual":
            model = build_visual_memory()
        elif name == "key_box":
            model = KeyBoxReader(
                build_model(
                    width=16,
                    state_model="belief",
                    memory_recent=2,
                    memory_block=2,
                    memory_blocks=1,
                ),
                EntityMatchReader(),
                EntityStateCell(preserve_no_information=True),
            )
        elif name == "interaction":
            model = EntityInteractionCell()
        elif name == "relation":
            model = RelationKey()
        elif name == "gate":
            model = RelationWriteGate()
        elif name == "perception":
            encoder = CNNEncoder(depth=0)
            model = Perception(
                encoder,
                dict(
                    rgb=ReconstructionDecoder(encoder.feature_spec),
                    pose=PushTPoseHead(encoder.feature_spec),
                ),
            )
        else:
            encoder = CNNEncoder(depth=0)
            model = WorldModel(
                encoder,
                MemoryUpdater(encoder.feature_spec, action_width=2, memory_width=128),
                Predictor(
                    encoder.feature_spec, action_width=2, memory_width=128, depth=2
                ),
            )
        raw = torch.load(path, map_location="cpu", weights_only=True)["model"]
        weights = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        model.load_state_dict(weights, strict=True)
        if file_hash(path) != sha:
            raise ValueError("Checkpoint changed during load")
        models[name] = model.eval().requires_grad_(False)
        receipts[name] = dict(
            path=str(path),
            sha256=sha,
            prefix=prefix,
            parameters=sum(p.numel() for p in model.parameters()),
            state_sha256=state_hash(model),
        )
    return models, settings, receipts


@torch.no_grad()
def capability_rollout(agent, data, settings, directory, name):
    """Observable free rollouts; future targets never enter the observation path."""
    from pathwm.evaluation.capabilities import prediction_metrics

    predicted, targets, currents, audios, audio_targets, actions_delta = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    generated, target_text, text_correct, source_rows = [], [], [], []
    for start in range(0, len(data), 2):
        ids = list(range(start, min(start + 2, len(data))))
        batch = data.batch(ids)
        torch.manual_seed(6801 + start)
        h, horizon = settings["history"], settings["horizon"]
        state = observe_history(agent, batch, h)
        a = batch["actions"][:, h - 1]
        torch.manual_seed(6901 + start)
        plus = agent.imagine(state, a)
        torch.manual_seed(6901 + start)
        minus = agent.imagine(state, -a)
        actions_delta.append(float((plus.tokens - minus.tokens).abs().mean()))
        future, sound = [], []
        for t in range(h, h + horizon):
            torch.manual_seed(7001 + start * horizon + t)
            state = agent.imagine(state, batch["actions"][:, t - 1])
            out = agent.decode(state)
            future.append(out["image"])
            sound.append(out["audio"])
        predicted.append(torch.stack(future, 1))
        targets.append(batch["images"][:, h : h + horizon])
        currents.append(batch["images"][:, h - 1])
        source_rows.extend(batch["sources"])
        if "audio" in batch:
            audios.append(torch.stack(sound, 1))
            audio_targets.append(batch["audio"][:, h : h + horizon])
        tokens = agent.generate_text(state, max_tokens=32).cpu().tolist()
        generated.extend(tokens)
        if "text" in batch:
            truth = batch["text"][:, h + horizon - 1].tolist()
            target_text.extend(truth)

            # BOS/PAD removed; compare content up to and including EOS.
            def content(row):
                r = [int(v) for v in row if v not in (0, 1)]
                return r[: r.index(2) + 1] if 2 in r else r

            text_correct.extend(content(a) == content(b) for a, b in zip(tokens, truth))
    predicted, targets, currents = map(torch.cat, (predicted, targets, currents))
    metrics = prediction_metrics(predicted, targets, currents)
    metrics.update(
        examples=len(data),
        action_token_mean_absolute_change=float(np.mean(actions_delta)),
    )
    arrays = dict(
        predicted=predicted.numpy(), target=targets.numpy(), current=currents.numpy()
    )
    if audios:
        sound, truth = torch.cat(audios), torch.cat(audio_targets)
        metrics.update(
            audio_mse=float((sound - truth).square().mean()),
            silence_audio_mse=float(truth.square().mean()),
        )
        arrays.update(audio_predicted=sound.numpy(), audio_target=truth.numpy())
    if text_correct:
        metrics["text_exact_match"] = float(np.mean(text_correct))
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    atomic_json(
        directory / f"{name}.json",
        dict(
            metrics=metrics,
            generated_token_ids=generated,
            target_token_ids=target_text,
            source_rows=source_rows,
            identity=data.identity,
            action_token_changes=actions_delta,
            arrays=f"{name}.npz",
        ),
    )
    return metrics


@torch.no_grad()
def capability_baseline(
    output,
    *,
    resume=False,
    check_only=False,
    overrides=None,
    reference=None,
    software=None,
):
    """Fixed current-capability scorecard; no training and no checkpoint selection."""
    import inspect
    import resource
    from PIL import Image
    from pathwm.data.visual_memory import VisualMemoryEpisodes
    from pathwm.evaluation.visual_memory import visual_features, visual_metrics
    from pathwm.evaluation.capabilities import (
        transform_visual,
        score,
        case_record,
        compare_baselines,
    )
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.data.entity_temporal import temporal_episodes
    from pathwm.evaluation.entity_state import state_metrics, state_runtime
    from pathwm.evaluation.entity_relations import evaluate_relations
    from pathwm.evaluation.source_choice import (
        evaluate_source_choice,
        evaluate_source_drift,
    )
    from pathwm.evaluation.key_box import evaluate_key_box
    from pathwm.models.key_box import plan_key
    from pathwm.data.images import PushTFrames

    if resume:
        saved_settings = json.loads((Path(output) / "run.json").read_text())[
            "identity"
        ]["settings"]
        overrides = (
            saved_settings["checkpoint_overrides"] if overrides is None else overrides
        )
    seed_everything(6001)
    models, model_settings, checkpoints = capability_models(overrides)
    if check_only:
        return dict(status="strict_checkpoint_loads_pass", checkpoints=checkpoints)
    started = perf_counter()
    deadline = started + 1200
    protocol = dict(
        schema="pathwm-capabilities-v1",
        seed=6001,
        pairs=32,
        visual_seed=6301,
        noise_seed=6401,
        rollout_count=16,
        temporal_seed=6501,
        relation_seed=6502,
        source_seed=6601,
        drift_seed=6651,
        planner_seed=6701,
        perturbations=[
            "original",
            "mirror",
            "swap_red_blue",
            "grayscale",
            "dim",
            "noise",
        ],
        math="MSE over every pixel/sample; paired accuracy; CE; original component probability/cost metrics",
        recipe_evaluator_sha256=digest(
            inspect.getsource(capability_baseline)
            + inspect.getsource(capability_rollout)
        ),
        evaluators={
            p.name: file_hash(p)
            for p in Path("pathwm/evaluation").glob("*.py")
            if p.name not in ("report.py", "diagrams.py")
        },
        settings={
            k: {
                x: v
                for x, v in s.items()
                if x not in ("device", "learning_rate", "steps", "seed")
            }
            for k, s in model_settings.items()
        },
    )
    media_root = Path("data/memory_media_v1")
    media = json.loads((media_root / "manifest.json").read_text())
    for r in media["records"]:
        if file_hash(media_root / r["path"]) != r["sha256"]:
            raise ValueError("Real-media source checksum mismatch")
    settings = dict(
        dataset="capabilities",
        device="cpu",
        seed=6001,
        purpose="diagnostic",
        protocol_sha256=digest(protocol),
        checkpoint_overrides=overrides or {},
        checkpoint_files=checkpoints,
        max_seconds=1200,
        optimizer_updates=0,
        baseline_reference=str(reference) if reference else None,
        capability_software=str(software) if software else None,
        software_sha256=file_hash(software) if software else None,
    )
    optimizer = torch.optim.SGD(
        models.parameters(), lr=0.0
    )  # Run schema only; never stepped.
    run = Run(
        output,
        settings=settings,
        data=dict(protocol=protocol, media=media),
        recipe=__file__,
        model=models,
        optimizer=optimizer,
        device="cpu",
        resume=resume,
    )
    if resume:
        result = json.loads((run.path / "capabilities.json").read_text())
        for filename, sha in json.loads(
            (run.path / "artifacts.json").read_text()
        ).items():
            if file_hash(run.path / filename) != sha:
                raise ValueError(f"Cached result changed: {filename}")
        if reference:
            atomic_json(
                run.path / "comparison.json",
                compare_baselines(
                    json.loads((Path(reference) / "capabilities.json").read_text()),
                    result,
                ),
            )
        run.status("completed", "pending")
        return render_report(run.path)
    if software:
        import shutil

        evidence = json.loads(Path(software).read_text())
        if any(file_hash(path) != sha for path, sha in evidence["sources"].items()):
            raise ValueError("Software-test source differs from evaluation source")
        shutil.copyfile(software, run.path / "software.json")
    before = state_hash(models)
    cases, gaps = [], []

    def record(name, role, inputs, metrics, gates=None, scope="", raw=None):
        if perf_counter() > deadline:
            raise TimeoutError("Capability evaluation budget exhausted")
        roles = role if isinstance(role, list) else [role]
        row = case_record(
            name,
            {k: checkpoints[k] for k in roles},
            inputs,
            metrics,
            gates or {},
            scope,
            raw or f"{name}.json",
        )
        cases.append(row)
        atomic_json(
            run.path / "capabilities.partial.json",
            dict(protocol_sha256=digest(protocol), cases=cases),
        )
        print(name, row["status"], json.dumps(metrics), flush=True)

    with evaluation_mode(models):
        # Every variant preserves the opposite-label paired final observation.
        data = VisualMemoryEpisodes(pairs=32, seed=6301)
        for condition in protocol["perturbations"]:
            changed = transform_visual(data, condition, seed=6401)
            full = visual_metrics(
                models["visual"].head(visual_features(models["visual"], changed)),
                changed,
            )
            erased = visual_metrics(
                models["visual"].head(
                    visual_features(models["visual"], changed, erased=True)
                ),
                changed,
            )
            if not torch.equal(
                torch.tensor(erased["logits"])[::2],
                torch.tensor(erased["logits"])[1::2],
            ):
                raise ValueError("Erased-history paired control differs")
            name = f"visual_{condition}"
            metrics = {k: full[k] for k in ("accuracy", "pair_both", "nll")}
            metrics.update(
                reversal_accuracy=full["reversal"]["accuracy"],
                erased_accuracy=erased["accuracy"],
                history_advantage=full["accuracy"] - erased["accuracy"],
            )
            gates = {
                k: score(metrics[k], v)
                for k, v in dict(
                    accuracy=0.9,
                    pair_both=0.8,
                    reversal_accuracy=0.8,
                    history_advantage=0.3,
                ).items()
            }
            atomic_json(
                run.path / f"{name}.json",
                dict(full=full, erased=erased, identity=changed.identity),
            )
            np.savez_compressed(
                run.path / f"{name}.npz", images=changed.images, labels=changed.labels
            )
            record(
                name,
                "visual",
                changed.identity,
                metrics,
                gates,
                "Fixed question, one target, four frames. Specific perturbation only; paired scenes are the sampling units.",
            )

        # Test real image input and actual decoder outputs of the constructed checkpoint.
        photos = [r for r in media["records"] if r["kind"] == "coco_image"]
        rgb = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        Image.open(media_root / r["path"])
                        .convert("RGB")
                        .resize((32, 32))
                    ).copy()
                )
                .permute(2, 0, 1)
                .float()
                / 255
                for r in photos
            ]
        )
        agent = models["visual"].agent
        torch.manual_seed(7101)
        state = agent.initial_state(len(rgb))
        state = agent.observe(
            state,
            {"image": Observation(rgb[:, None], torch.zeros(len(rgb), 1))},
            time=0,
        )
        outputs = agent.decode(state)
        tokens = agent.generate_text(state)
        torch.manual_seed(7201)
        plus = agent.imagine(state, torch.ones(len(rgb), 2))
        torch.manual_seed(7201)
        minus = agent.imagine(state, -torch.ones(len(rgb), 2))
        metric = dict(
            image_mse=float((outputs["image"] - rgb).square().mean()),
            gray_image_mse=float((0.5 - rgb).square().mean()),
            image_output_std=float(outputs["image"].std(unbiased=False)),
            audio_output_rms=float(outputs["audio"].square().mean().sqrt()),
            generated_content_tokens=int((tokens > 2).sum()),
            action_token_change=float((plus.tokens - minus.tokens).abs().mean()),
        )
        modality_arrays = {}
        for modality in ("image", "video", "audio", "text"):
            if modality == "text":
                values, valid = bytes_batch(["left", "rght"])
                obs = Observation(
                    values, torch.zeros_like(values, dtype=torch.float32), valid
                )
            elif modality == "audio":
                values = torch.stack((torch.zeros(1, 32), torch.ones(1, 32)))
                obs = Observation(values, torch.zeros(2, 1))
            else:
                values = torch.zeros(2, 1, 3, 32, 32)
                values[1, :, 0] = 1
                obs = Observation(values, torch.zeros(2, 1))
            encoded = (
                agent.encode(agent.initial_state(2), {modality: obs}, time=0)[modality]
                .as_tokens()
                .values
            )
            metric[f"{modality}_input_feature_change"] = float(
                (encoded[0] - encoded[1]).abs().mean()
            )
            modality_arrays[modality] = encoded.numpy()
        np.savez_compressed(
            run.path / "constructed_input_features.npz", **modality_arrays
        )
        counts = {
            name: dict(
                parameters=sum(p.numel() for p in m.parameters()),
                nonzero=sum(int(p.count_nonzero()) for p in m.parameters()),
            )
            for name, m in agent.named_children()
        }
        np.savez_compressed(
            run.path / "constructed_modalities.npz",
            input=rgb.numpy(),
            image=outputs["image"].numpy(),
            audio=outputs["audio"].numpy(),
            text=tokens.numpy(),
            video=agent.decode_video([state, plus]).numpy(),
        )
        atomic_json(
            run.path / "constructed_modalities.json",
            dict(
                metrics=metric,
                parameter_counts=counts,
                generated_token_ids=tokens.tolist(),
                photos=photos,
            ),
        )
        record(
            "constructed_modalities",
            "visual",
            photos,
            metric,
            dict(
                image_beats_gray=score(
                    metric["image_mse"],
                    metric["gray_image_mse"],
                    lower=True,
                    strict=True,
                )
            ),
            "Real COCO development photographs, image/audio/text/video output and action sensitivity. No semantic memory labels.",
        )

        for name in ("synthetic", "instructions", "pusht"):
            s = dict(
                model_settings[name],
                validation_windows=16,
                test_windows=16,
                device="cpu",
                batch_size=2,
            )
            data = make_data(s, "test")
            metrics = capability_rollout(
                models[name], data, s, run.path, f"prediction_{name}"
            )
            gates = dict(
                image_beats_copy=score(
                    metrics["image_mse"],
                    metrics["copy_image_mse"],
                    lower=True,
                    strict=True,
                )
            )
            if "audio_mse" in metrics:
                gates["audio_beats_silence"] = score(
                    metrics["audio_mse"],
                    metrics["silence_audio_mse"],
                    lower=True,
                    strict=True,
                )
            record(
                f"prediction_{name}",
                name,
                data.identity,
                metrics,
                gates,
                "Short-pilot weights; free rollout, fixed test split. PushT windows overlap within an episode. Sensitivity is not correctness.",
            )
            if name == "instructions":
                task_metrics, task_rows = task_evaluation(models[name], data, s)
                atomic_json(
                    run.path / "instructions.json",
                    dict(metrics=task_metrics, rows=task_rows),
                )
                record(
                    "instructions",
                    name,
                    data.identity,
                    task_metrics,
                    dict(
                        operation=score(1 - task_metrics["task_operation_error"], 0.9)
                    ),
                    "Finite operation templates and enforced output contracts; not general dialogue or tool execution.",
                )
        for role in ("facts_direct", "facts_event"):
            s = dict(model_settings[role], device="cpu")
            data = make_data(s, "validation")
            raw = fact_predictions(models[role], data, s, deadline=deadline)
            metrics = fact_metrics(**raw)
            atomic_json(
                run.path / f"{role}.json", {k: v.tolist() for k, v in raw.items()}
            )
            record(
                role,
                role,
                data.identity,
                metrics,
                {
                    k: score(metrics[k], 0.9)
                    for k in ("entity_accuracy", "location_accuracy")
                },
                "Reused 32 development combinations of predefined IDs; regression, not new visual identity evidence.",
            )
        s = dict(model_settings["recall"], device="cpu")
        data = make_data(s, "test")
        logits, labels, examples = recall_predictions(
            models["recall"], data, s, deadline=deadline
        )
        metric = recall_metrics(logits, labels)
        groups = {
            g: recall_metrics(
                logits[torch.tensor([r["group"] == g for r in examples])],
                labels[torch.tensor([r["group"] == g for r in examples])],
            )
            for g in sorted({r["group"] for r in examples})
        }
        atomic_json(
            run.path / "recall.json",
            dict(
                logits=logits.tolist(),
                labels=labels.tolist(),
                examples=examples,
                groups=groups,
            ),
        )
        record(
            "recall",
            "recall",
            data.identity,
            metric,
            dict(
                beats_abstain=score(metric["task_loss"], 0.25, lower=True, strict=True)
            ),
            "Existing test population reused; development-selected target checkpoint; raw temperature1, no new fitting.",
        )

        key = models["key_box"]
        from pathwm.evaluation.entities import matching_metrics

        novelty = VariableEntityMatches("validation", 256)
        scores = entity_predictions(
            key.matcher, novelty, dict(batch_size=32, device="cpu"), deadline=deadline
        )
        raw = matching_metrics(scores["logits"], scores["targets"], novelty.cohorts)
        logits = scores["logits"][0]
        target = scores["targets"][0].argmax(-1)
        metric = dict(accuracy=float((logits.argmax(-1) == target).float().mean()))
        for cohort in ("known", "novel"):
            mask = torch.tensor([x == cohort for x in novelty.cohorts])
            metric[cohort + "_accuracy"] = float(
                (logits.argmax(-1)[mask] == target[mask]).float().mean()
            )
        atomic_json(
            run.path / "entity_novelty.json",
            dict(
                scores=raw,
                logits=logits.tolist(),
                labels=target.tolist(),
                manifest=novelty.manifest,
            ),
        )
        record(
            "entity_novelty",
            "key_box",
            novelty.identity,
            metric,
            dict(
                known=score(metric["known_accuracy"], 0.95),
                novel=score(metric["novel_accuracy"], 0.95),
            ),
            "Reused variable-candidate development descriptors. Raw false merges/splits and abstention coverage retained; not visual discovery.",
        )
        families = growth_inputs(6501, 8)
        populations = temporal_episodes(key.matcher, families, idle_lengths=(3, 31))
        # Direct matching makes identity ambiguity visible independently of state readout.
        query = torch.tensor([f["revisits"][i] for f in families for i in range(2)])
        memory = torch.tensor(
            [f["descriptors"][:2] for f in families for _ in range(2)]
        )
        matching = key.matcher.match(query, memory)
        labels = torch.tensor([0, 1] * len(families))
        metric = dict(
            accuracy=float((matching.argmax(-1) == labels).float().mean()),
            examples=len(labels),
        )
        atomic_json(
            run.path / "entity_matching.json",
            dict(logits=matching.tolist(), labels=labels.tolist(), families=families),
        )
        record(
            "entity_matching",
            "key_box",
            families,
            metric,
            dict(accuracy=score(metric["accuracy"], 0.95)),
            "Supplied descriptors and known two-entity candidates; novelty/ambiguous matching remains covered by specialist software checks.",
        )
        for condition, data in populations.items():
            metric, logits, _ = state_metrics(key.cell, data)
            runtime = state_runtime(key.cell, key.matcher, data, deadline)
            metric.update(
                runtime_pair_accuracy=runtime["pair_accuracy"],
                transactions=runtime["transactions"],
                latent_agreement=runtime["latent_agreement"],
            )
            name = f"state_{condition}"
            atomic_json(
                run.path / f"{name}.json",
                dict(
                    logits=logits.tolist(),
                    targets=data["targets"].tolist(),
                    runtime=runtime,
                ),
            )
            record(
                name,
                "key_box",
                {k: v for k, v in data.items() if k != "slots"},
                metric,
                dict(pair_accuracy=score(metric["pair_accuracy"], 0.95)),
                "Learned state cell with supplied operation vectors; explicit identity records and no-information preservation.",
            )
        families = growth_inputs(6502, 2)
        relation = evaluate_relations(
            key.matcher, models["interaction"], models["relation"], families, deadline
        )
        for condition, raw in relation["cohorts"].items():
            name = f"relation_{condition}"
            atomic_json(run.path / f"{name}.json", raw)
            metric = {k: v for k, v in raw.items() if isinstance(v, (float, int, bool))}
            record(
                name,
                ["key_box", "interaction", "relation"],
                dict(families=families, condition=condition),
                metric,
                dict(accuracy=score(metric["accuracy"], 0.95))
                if condition != "erased"
                else {},
                scope="One supplied relation type; learned addressing and state interaction, not learned graph topology.",
            )
        for name, raw in (
            ("source_static", evaluate_source_choice(models["gate"], 16, seed=6601)),
            ("source_drift", evaluate_source_drift(models["gate"], 16, seed=6651)),
        ):
            atomic_json(run.path / f"{name}.json", raw)
            if name == "source_static":
                metric = dict(
                    useful_source_rate=raw["useful_source_rate"],
                    utility=raw["summary"]["learned"]["utility"],
                    stop_utility=raw["summary"]["stop"]["utility"],
                    accuracy=raw["summary"]["learned"]["accuracy"],
                )
                gates = dict(
                    beats_stop=score(
                        metric["utility"], metric["stop_utility"], strict=True
                    )
                )
            else:
                metric = {
                    f"{condition}_{policy}_{k}": v
                    for condition, policies in raw["summary"].items()
                    for policy, metrics in policies.items()
                    for k, v in metrics.items()
                }
                gates = {}
            record(
                name,
                "gate",
                dict(seed=6601 if name == "source_static" else 6651, worlds=16),
                metric,
                gates,
                "Outcome-feedback adaptation with supplied sources; no neural reliability estimator. Raw source feedback retained.",
            )
        for stress in (False, True):
            raw = evaluate_key_box(key, families=16, seed=6701, stress=stress)
            name = "planning_relocation" if stress else "planning_ordinary"
            atomic_json(run.path / f"{name}.json", raw)
            metric = dict(
                raw["summary"]["integrated"],
                known_accuracy=raw["known_accuracy"],
                no_history_utility=raw["summary"]["no_history"]["utility"],
                supplied_state_success=raw["summary"]["supplied_state"]["success"],
            )
            if raw["correction_accuracy"] is not None:
                metric["correction_accuracy"] = raw["correction_accuracy"]
            record(
                name,
                "key_box",
                dict(families=growth_inputs(6701, 16), stress=stress),
                metric,
                dict(
                    success=score(metric["success"], 0.95),
                    absent=score(metric["absent_stop"], 0.95),
                ),
                "Supplied expectimax mechanics, four-action horizon, learned workspace reading. No learned world-model planner.",
            )
        action, value = plan_key((0.15, 0, 0.85), (True, False), 1)
        selected = (
            0.1
            if action == ("retrieve", 0)
            else 0.85
            if action == ("stop", -1)
            else None
        )
        raw = dict(
            belief=[0.15, 0, 0.85],
            opened=[True, False],
            horizon=1,
            action=action,
            planner_value=value,
            reported_expected_utility=selected,
            stop_expected_utility=0.85,
        )
        atomic_json(run.path / "planner_objective.json", raw)
        record(
            "planner_objective",
            "key_box",
            dict(belief=raw["belief"], opened=raw["opened"], horizon=1),
            dict(selected_utility=selected, best_utility=0.85),
            dict(alignment=score(selected, 0.85)),
            "Previously known surrogate-reward mismatch, retained unchanged for the baseline.",
        )

        # The separately trained CNN/temporal references use their own compatible weights.
        frames = PushTFrames("data/pusht_world_model/cchi_v1", split="test")
        batch = frames.batch(range(16))
        frame_identity = dict(
            frames.identity,
            evaluated_rows=frames.rows[:16].tolist(),
            evaluated_count=16,
        )
        predicted = models["perception"](batch["rgb"])
        if isinstance(predicted, tuple):
            predicted = predicted[0]
        metric = dict(
            rgb_mse=float((predicted["rgb"] - batch["rgb"]).square().mean()),
            gray_mse=float((0.5 - batch["rgb"]).square().mean()),
            pose_mse=float((predicted["pose"] - batch["pose"]).square().mean()),
        )
        np.savez_compressed(
            run.path / "perception.npz",
            input=batch["rgb"].numpy(),
            predicted=predicted["rgb"].numpy(),
            pose=predicted["pose"].numpy(),
            target_pose=batch["pose"].numpy(),
        )
        atomic_json(
            run.path / "perception.json", dict(metrics=metric, identity=frame_identity)
        )
        record(
            "perception",
            "perception",
            frame_identity,
            metric,
            dict(
                rgb=score(
                    metric["rgb_mse"], metric["gray_mse"], lower=True, strict=True
                )
            ),
            "Separate short-trained CNN reconstruction/pose reference. Grouped test-frame prefix.",
        )
        sequences = PushTSequences(
            "data/pusht_world_model/cchi_v1", "test", history=2, horizon=3, limit=16
        )
        predictions, targets, currents = [], [], []
        for start in range(0, 16, 2):
            batch = sequences.batch(range(start, start + 2))
            states = models["dynamics"](
                batch["history"],
                batch["history_actions"],
                batch["actions"],
                batch["initial_previous_action"],
            )
            predictions.append(
                torch.stack(
                    [models["perception"].heads["rgb"](x[0]) for x in states], 1
                )
            )
            targets.append(batch["future"])
            currents.append(batch["history"][:, -1])
        from pathwm.evaluation.capabilities import prediction_metrics

        p, t, c = map(torch.cat, (predictions, targets, currents))
        metric = prediction_metrics(p, t, c)
        np.savez_compressed(
            run.path / "dynamics.npz",
            predicted=p.numpy(),
            target=t.numpy(),
            current=c.numpy(),
        )
        atomic_json(
            run.path / "dynamics.json",
            dict(metrics=metric, identity=sequences.identity),
        )
        record(
            "dynamics",
            ["dynamics", "perception"],
            sequences.identity,
            metric,
            dict(
                beats_copy=score(
                    metric["image_mse"],
                    metric["copy_image_mse"],
                    lower=True,
                    strict=True,
                )
            ),
            "Separate temporal reference, paired diagnostic decoder; overlapping real test windows.",
        )

    for name, reason in [
        (
            "real_visual_identity",
            "No trained pixels-to-entity candidate/matching interface or reviewed real temporal identity answers.",
        ),
        (
            "faces_and_person_beliefs",
            "No current trained face recognition or person-state evaluation interface.",
        ),
        (
            "learned_graph_and_concepts",
            "Records and a supplied relation exist; learned topology/concept/skill formation is not implemented.",
        ),
        (
            "speech_and_conversation",
            "Toy waveform and byte decoders exist; no usable speech or conversational training/evaluation contract.",
        ),
        (
            "essays_and_general_writing",
            "No general language model/checkpoint or grounded writing evaluator.",
        ),
        (
            "goal_directed_image_video",
            "Reconstruction/imagined frames are measured above; prompt-conditioned content generation is not implemented.",
        ),
        (
            "general_software_tools",
            "Task proposals/output contracts exist; no integrated external software executor and verified completion loop.",
        ),
        (
            "learned_dynamics_planning",
            "Learned prediction and supplied-mechanics planning remain separate; no combined learned controller.",
        ),
        (
            "live_webcam_memory",
            "Camera/import software tested separately; live camera and reviewed real memory labels are not in this run.",
        ),
        (
            "whole_agent_gpu_budget",
            "This run measures CPU inference. Previous narrow visual GPU test cannot establish full training/inference headroom.",
        ),
    ]:
        gaps.append(dict(id=name, status="not_measured", reason=reason))
    if state_hash(models) != before:
        raise ValueError("Evaluation mutated checkpoint tensors")
    result = dict(
        schema="pathwm-capabilities-v1",
        protocol=protocol,
        protocol_sha256=digest(protocol),
        cases=cases,
        gaps=gaps,
        checkpoints=checkpoints,
        resources=dict(
            cpu_wall_seconds=perf_counter() - started,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        ),
        limits=[
            "Separate checkpoints, not one multi-capability agent.",
            "Frozen small screens, no training or checkpoint selection; finite robustness only.",
            "Failures describe these evaluated paths, not proof that latent information is absent.",
            "Raw artifacts support audit; software passing is separate from learned performance.",
        ],
    )
    atomic_json(run.path / "capabilities.json", result)
    if reference:
        atomic_json(
            run.path / "comparison.json",
            compare_baselines(
                json.loads((Path(reference) / "capabilities.json").read_text()), result
            ),
        )
    run.save()
    atomic_json(
        run.path / "artifacts.json",
        {
            p.name: file_hash(p)
            for p in run.path.iterdir()
            if p.suffix in (".json", ".npz", ".pt")
            and p.name not in ("status.json", "artifacts.json")
        },
    )
    run.status("completed", "pending")
    return render_report(run.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-weight-timing",
        action="store_true",
        help="Use the predeclared numeric timing revision and fresh test scenes",
    )
    parser.add_argument(
        "--entity-association", choices=["raw", "observed", "learned"], default="raw"
    )
    parser.add_argument(
        "--entity-reader", choices=["recurrent", "shared"], default="recurrent"
    )
    parser.add_argument("--entity-noise", type=float, default=0.0)
    parser.add_argument(
        "--capability-software",
        type=Path,
        help="Explicit full-suite verification JSON to retain beside behavioral results",
    )
    parser.add_argument(
        "--capability-weights",
        type=Path,
        help="JSON role-to-checkpoint overrides for a frozen capability comparison",
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        help="Compare with a run of the identical capability protocol",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--diagram",
        type=Path,
        nargs="?",
        const=Path("docs/diagrams"),
        help="Export Gaussian reference diagrams to this directory (CPU, no training)",
    )
    parser.add_argument("--diagram-depth", type=int, default=2)
    parser.add_argument("--output", default="runs/multimodal_first")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--dataset",
        choices=[
            "capabilities",
            "synthetic",
            "instructions",
            "pusht",
            "recall",
            "facts",
            "entities",
            "entity-matching",
            "direct-weights",
        ],
        default="synthetic",
    )
    parser.add_argument(
        "--state-model", choices=["belief", "gaussian"], default="belief"
    )
    parser.add_argument(
        "--entity-growth-weights",
        help="Frozen entity matcher checkpoint for growth evaluation",
    )
    parser.add_argument(
        "--entity-state-weights",
        help="Frozen recognizer checkpoint for learned state binding",
    )
    parser.add_argument(
        "--entity-temporal-cell", help="Frozen state checkpoint for temporal evaluation"
    )
    parser.add_argument("--entity-source-drift", action="store_true")
    parser.add_argument("--entity-source-choice", action="store_true")
    parser.add_argument("--entity-evidence-sources", action="store_true")
    parser.add_argument("--entity-gate-correlation", type=float)
    parser.add_argument("--entity-gate-reobserve", action="store_true")
    parser.add_argument("--entity-gate-shift", action="store_true")
    parser.add_argument("--entity-gate-weights", type=Path)
    parser.add_argument("--entity-gate-retain", type=float, default=0.0)
    parser.add_argument("--entity-gate-replicate", type=int, default=0)
    parser.add_argument("--entity-gate-augment", action="store_true")
    parser.add_argument("--entity-gate", action="store_true")
    parser.add_argument("--entity-relation-key", type=Path)
    parser.add_argument("--entity-relations", action="store_true")
    parser.add_argument("--entity-source", action="store_true")
    parser.add_argument("--entity-interaction", action="store_true")
    parser.add_argument("--entity-interaction-blind", action="store_true")
    parser.add_argument("--entity-state-varied", action="store_true")
    parser.add_argument("--entity-state-preserve", action="store_true")
    parser.add_argument("--entity-temporal-idle", action="store_true")
    parser.add_argument("--entity-temporal-seed", type=int, default=111)
    parser.add_argument("--entity-temporal-varied", action="store_true")
    parser.add_argument("--entity-variable", action="store_true")
    parser.add_argument("--entity-growth-seed", type=int, default=61)
    parser.add_argument("--fact-reader", choices=["direct", "event"], default="direct")
    parser.add_argument(
        "--fact-encoder-weights",
        help="Direct-fact checkpoint: load only agent.encoder into the trainable event reader",
    )
    parser.add_argument("--memory-recent", type=int, default=32)
    parser.add_argument("--memory-block", type=int, default=8)
    parser.add_argument("--memory-blocks", type=int, default=16)
    parser.add_argument(
        "--recall-mode", choices=["history", "current-recent"], default="history"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Cumulative active-time cap for recall or direct fact diagnostics",
    )
    parser.add_argument(
        "--recall-truncate",
        type=int,
        default=32,
        help="Recall gradient segment length; zero retains the complete graph",
    )
    parser.add_argument("--calibration-windows", type=int, default=15)
    parser.add_argument("--test-windows", type=int, default=15)
    parser.add_argument("--abstain-cost", type=float, default=0.25)
    parser.add_argument("--data-root", default="data/pusht_world_model/cchi_v1")
    for name, default in [
        ("steps", 8),
        ("batch-size", 2),
        ("width", 32),
        ("image-size", 16),
        ("audio-samples", 32),
        ("history", 2),
        ("horizon", 2),
        ("train-windows", 32),
        ("validation-windows", 8),
        ("evaluate-every", 4),
        ("seed", 42),
        ("improve-every", 4),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.diagram is not None and (
        args.check
        or args.resume
        or args.stop_after is not None
        or args.device != "cpu"
        or args.dataset != "synthetic"
    ):
        parser.error(
            "--diagram uses a fresh synthetic CPU example; run it separately from training/check/resume"
        )
    if args.diagram_depth < 0:
        parser.error("Diagram depth must be nonnegative")
    args = resume_arguments(parser, args)
    if args.dataset == "capabilities":
        if args.device != "cpu" or args.stop_after is not None:
            parser.error(
                "Capability baseline uses CPU evaluation without optimizer steps"
            )
        overrides = (
            json.loads(args.capability_weights.read_text())
            if args.capability_weights
            else None
        )
        print(
            capability_baseline(
                args.resume or args.output,
                resume=bool(args.resume),
                check_only=args.check,
                overrides=overrides,
                reference=args.baseline_reference,
                software=args.capability_software,
            )
        )
        return
    if args.dataset == "direct-weights":
        if args.check or args.stop_after is not None:
            parser.error(
                "Direct-weight construction is an evaluation recipe; use its focused tests for checks"
            )
        print(
            direct_visual_weights(
                args.resume or args.output,
                device=args.device,
                resume=args.resume is not None,
                timing=args.direct_weight_timing,
            )
        )
        return
    if args.entity_source_drift:
        if args.entity_gate_weights is None:
            parser.error("--entity-source-drift requires --entity-gate-weights")
        print(
            evaluate_entity_source_drift(
                args.entity_gate_weights, args.resume or args.output, bool(args.resume)
            )
        )
        return
    if args.entity_source_choice:
        if args.entity_gate_weights is None:
            parser.error("--entity-source-choice requires --entity-gate-weights")
        print(
            evaluate_entity_source_choice(
                args.entity_gate_weights, args.resume or args.output, bool(args.resume)
            )
        )
        return
    if args.entity_evidence_sources:
        if args.entity_gate_weights is None:
            parser.error("--entity-evidence-sources requires --entity-gate-weights")
        print(
            evaluate_entity_evidence_sources(
                args.entity_gate_weights, args.resume or args.output, bool(args.resume)
            )
        )
        return
    if args.entity_gate_shift:
        if args.entity_gate_weights is None:
            parser.error("--entity-gate-shift requires --entity-gate-weights")
        print(
            evaluate_entity_gate_shift(
                args.entity_gate_weights,
                args.resume or args.output,
                bool(args.resume),
                reobserve=args.entity_gate_reobserve,
                correlation=args.entity_gate_correlation,
            )
        )
        return
    if args.entity_gate:
        if any(
            v is None
            for v in (
                args.entity_state_weights,
                args.entity_temporal_cell,
                args.entity_relation_key,
            )
        ):
            parser.error(
                "--entity-gate requires matcher, interaction and relation key checkpoints"
            )
        print(
            train_entity_gate(
                args.entity_state_weights,
                args.entity_temporal_cell,
                args.entity_relation_key,
                args.resume or args.output,
                bool(args.resume),
                gate_weights=args.entity_gate_weights,
                augmented=args.entity_gate_augment,
                replicate=args.entity_gate_replicate,
                retention=args.entity_gate_retain,
            )
        )
        return
    if args.entity_relations:
        if args.entity_state_weights is None or args.entity_temporal_cell is None:
            parser.error(
                "--entity-relations requires recognition and interaction checkpoints"
            )
        print(
            train_entity_relations(
                args.entity_state_weights,
                args.entity_temporal_cell,
                args.resume or args.output,
                bool(args.resume),
            )
        )
        return
    if args.entity_source:
        if args.entity_state_weights is None or args.entity_temporal_cell is None:
            parser.error(
                "--entity-source requires recognition and interaction checkpoints"
            )
        print(
            evaluate_entity_source(
                args.entity_state_weights,
                args.entity_temporal_cell,
                args.resume or args.output,
                bool(args.resume),
            )
        )
        return
    if args.entity_interaction:
        if args.entity_state_weights is None or args.entity_temporal_cell is None:
            parser.error(
                "--entity-interaction requires recognition and state checkpoints"
            )
        print(
            train_entity_interaction(
                args.entity_state_weights,
                args.entity_temporal_cell,
                args.resume or args.output,
                bool(args.resume),
                args.entity_interaction_blind,
            )
        )
        return
    if args.entity_temporal_cell is not None:
        if args.entity_state_weights is None:
            parser.error("--entity-temporal-cell requires --entity-state-weights")
        print(
            evaluate_entity_temporal(
                args.entity_state_weights,
                args.entity_temporal_cell,
                args.resume or args.output,
                bool(args.resume),
                seed=args.entity_temporal_seed,
                varied=args.entity_temporal_varied,
                idle=args.entity_temporal_idle,
            )
        )
        return
    if args.entity_state_weights is not None:
        print(
            train_entity_state(
                args.entity_state_weights,
                args.resume or args.output,
                bool(args.resume),
                varied=args.entity_state_varied,
                preserve=args.entity_state_preserve,
            )
        )
        return
    if args.entity_growth_weights is not None:
        print(
            entity_growth(
                args.entity_growth_weights,
                args.resume or args.output,
                bool(args.resume),
                seed=args.entity_growth_seed,
            )
        )
        return
    if args.fact_encoder_weights is not None and (
        args.dataset != "facts" or args.fact_reader != "event"
    ):
        parser.error(
            "--fact-encoder-weights requires --dataset facts --fact-reader event"
        )
    if args.fact_reader != "direct" and (
        args.dataset != "facts" or args.state_model != "belief"
    ):
        parser.error("--fact-reader event requires --dataset facts and belief state")
    diagnostic = args.dataset in ("facts", "entities", "entity-matching") or (
        args.dataset == "recall" and args.recall_mode == "current-recent"
    )
    if not np.isfinite(args.entity_noise) or not 0 <= args.entity_noise < 0.25:
        parser.error("Entity noise must be finite in [0, 0.25)")
    if args.entity_noise and (
        args.dataset != "entities" or args.entity_association != "learned"
    ):
        parser.error("Entity noise requires entities and learned association")
    if args.entity_association == "learned" and args.entity_reader != "shared":
        parser.error("Learned association requires shared reader")
    if args.entity_reader == "shared" and (
        args.dataset != "entities"
        or args.entity_association not in ("observed", "learned")
    ):
        parser.error("Shared reader requires entities and observed association")
    if args.entity_association != "raw" and args.dataset != "entities":
        parser.error("--entity-association requires --dataset entities")
    if args.recall_mode != "history" and args.dataset != "recall":
        parser.error("--recall-mode requires --dataset recall")
    if args.dataset in ("entities", "entity-matching"):
        import sys

        supplied = {v.split("=", 1)[0] for v in sys.argv[1:] if v.startswith("--")}
        if args.resume is None:
            for name, value in dict(
                width=64,
                history=3,
                horizon=1,
                train_windows=512,
                validation_windows=256,
                steps=256,
                batch_size=32,
                evaluate_every=32,
                seed=31,
                learning_rate=0.003,
                improve_every=0,
                max_seconds=450.0,
            ).items():
                if "--" + name.replace("_", "-") not in supplied:
                    setattr(args, name, value)
        if args.device != "cpu" or args.improve_every:
            parser.error("Entities require CPU and no extra-update gate")
    if args.dataset == "facts":
        import sys

        supplied = {
            arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        if args.resume is None:
            for name, value in (
                ("history", 1),
                ("horizon", 1),
                ("train_windows", 96),
                ("validation_windows", 32),
                ("steps", 512),
                ("batch_size", 16),
                ("evaluate_every", 32),
                ("seed", 23),
                ("improve_every", 0),
                ("max_seconds", 450.0),
            ):
                if "--" + name.replace("_", "-") not in supplied:
                    setattr(args, name, value)
        if args.device != "cpu" or args.improve_every or args.max_seconds <= 0:
            parser.error(
                "Facts require CPU, improve-every 0 and a positive active-time cap"
            )
    if (
        not np.isfinite(args.max_seconds)
        or args.max_seconds < 0
        or (args.max_seconds and not diagnostic)
    ):
        parser.error("--max-seconds is a nonnegative diagnostic-only budget")
    if args.dataset == "recall":
        import sys

        supplied = {
            arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        if args.resume is None:
            defaults = (
                ("history", 256),
                ("train_windows", 15),
                ("validation_windows", 15),
                ("improve_every", 0),
            )
            if diagnostic:
                defaults = (
                    ("history", 4),
                    ("train_windows", 40),
                    ("validation_windows", 100),
                    ("improve_every", 0),
                    ("memory_recent", 2),
                    ("memory_block", 2),
                    ("memory_blocks", 2),
                    ("recall_truncate", 4),
                    ("steps", 256),
                    ("batch_size", 4),
                    ("evaluate_every", 16),
                    ("seed", 17),
                    ("max_seconds", 900.0),
                )
            for name, value in defaults:
                if "--" + name.replace("_", "-") not in supplied:
                    setattr(args, name, value)
        if (
            args.state_model != "belief"
            or args.improve_every
            or args.recall_truncate < 0
            or any(
                n < 1 or n % (10 if diagnostic else 15)
                for n in (
                    (args.train_windows, args.validation_windows)
                    if diagnostic
                    else (
                        args.train_windows,
                        args.validation_windows,
                        args.calibration_windows,
                        args.test_windows,
                    )
                )
            )
            or not np.isfinite(args.abstain_cost)
            or args.abstain_cost < 0
        ):
            parser.error(
                "Recall requires belief, improve-every 0, nonnegative truncation/cost; counts divide by 10 for diagnostics, 15 for history"
            )
        if diagnostic and (args.max_seconds <= 0 or args.device != "cpu"):
            parser.error(
                "Recall diagnostic requires CPU and a positive --max-seconds budget"
            )
    counts = [
        args.steps,
        args.batch_size,
        args.width,
        args.image_size,
        args.audio_samples,
        args.history,
        args.horizon,
        args.train_windows,
        args.validation_windows,
        args.evaluate_every,
        args.memory_recent,
        args.memory_block,
        args.memory_blocks,
    ]
    if (
        min(counts) < 1
        or args.width % 4
        or args.image_size % 4
        or args.improve_every < 0
    ):
        parser.error(
            "Counts must be positive; width and image size must divide by 4; improve-every may be zero"
        )
    if (
        not 0 <= args.ema_decay < 1
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Learning rate must be positive and EMA decay in [0,1)")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("Stop-after must be positive")
    if args.diagram is not None:
        print(
            export_diagrams(
                args.diagram,
                depth=args.diagram_depth,
                width=args.width,
                image_size=args.image_size,
                audio_samples=args.audio_samples,
                seed=args.seed,
            )
        )
        return
    settings = {
        k: v
        for k, v in vars(args).items()
        if k
        not in ("check", "output", "resume", "stop_after", "diagram", "diagram_depth")
    }
    settings.update(
        purpose="development",
        precision="fp32",
        time_unit="one dataset transition",
        objective=(
            "categorical split KL; image/audio Gaussian sigma .1; text categorical; two-sample log-mean likelihood per modality per dimension; biased straight-through gradients; partial-view teacher; bounded memory replay/distillation/mark utility"
            if args.state_model == "belief"
            else "data reconstruction + future outputs + EMA latent Gaussian NLL + action NLL + error supervision; instructions adds operation CE + modality BCE + completion BCE"
        ),
        proposal_budget="one extra update per improve-every interval, rolled back on rejection",
    )
    if args.dataset == "recall":
        settings.update(
            objective="all-query factual CE + final-segment text/source grounding and split categorical KL; uniform episode sampling; development NLL checkpoint selection; independent scalar calibration",
            proposal_budget="none",
            time_unit="one ordered delivered text record",
        )
        if diagnostic:
            settings.update(
                objective="all-query factual CE + text/source grounding and split categorical KL; final checkpoint; train/development diagnostic only; no calibration",
                purpose="diagnostic",
            )
    if args.dataset == "facts":
        settings.update(
            purpose="diagnostic",
            objective=(
                "equal-weight entity/location CE through one committed agent event, task interpreter and working-token readout; final checkpoint only; fixed batch evaluation seeds"
                if args.fact_reader == "event"
                else "equal-weight entity/location CE from one observation; end-to-end source encoder; final checkpoint only"
            ),
            proposal_budget="none",
            time_unit="single observation at neutral time zero",
        )
    if args.dataset in ("entities", "entity-matching"):
        settings.update(
            purpose="diagnostic",
            objective="equal-weight identity/state-pair/post-action proper CE; recurrent controlled proposals; final checkpoint",
            proposal_budget="none",
            time_unit="one controlled observed event",
        )
    if args.dataset == "entity-matching":
        settings["objective"] = (
            "known-memory-or-new CE with dataset-defined candidate count; final checkpoint"
        )
    if args.check:
        print(json.dumps(check(settings), indent=2))
    else:
        print(
            train(
                settings,
                args.resume or args.output,
                resume=args.resume is not None,
                stop_after=args.stop_after,
            )
            / "report.html"
        )


if __name__ == "__main__":
    main()
