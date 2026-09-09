"""One explicit multimodal composition. No hidden recurrent state or registry."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .agent_state import LatentState
from .modalities import Attend, Observation, position, observation_values, bytes_batch
from .multiscale import FeaturePyramid
from .tasks import Provenance, GeneratedOutput, Emission, TaskStep


def batch_time(value, reference):
    # Keep long-running clocks precise even when activations use FP32/FP16.
    result = torch.as_tensor(value, device=reference.device, dtype=torch.float64)
    if result.ndim == 0:
        result = result.expand(len(reference))
    if result.shape != (len(reference),) or not torch.isfinite(result).all():
        raise ValueError("Time must be finite scalar or [B]")
    return result


def record_state(trace, name, state):
    if trace is not None:
        trace[name + ".tokens"] = state.tokens.detach().cpu().clone()
        trace[name + ".log_scale"] = state.log_scale.detach().cpu().clone()
        trace[name + ".time"] = state.time.detach().cpu().clone()
        trace[name + ".imagined"] = state.imagined


class ObservationUpdate(nn.Module):
    def __init__(self, width, action_width=2, depth=1):
        super().__init__()
        self.action_width = action_width
        self.action = nn.Linear(action_width + 1, width)
        self.blocks = nn.ModuleList([Attend(width) for _ in range(depth)])
        self.scale = nn.Linear(width, width)

    def forward(self, state, tokens, valid, previous_action, now, trace=None):
        b = len(tokens)
        present = tokens.new_full((b, 1), float(previous_action is not None))
        action = (
            tokens.new_zeros(b, self.action_width)
            if previous_action is None
            else previous_action
        )
        if action.shape != (b, self.action_width) or not torch.isfinite(action).all():
            raise ValueError("Previous action must be finite [B,action_width]")
        condition = self.action(torch.cat((action, present), -1))
        condition = condition + position(now - state.time, tokens.shape[-1]).to(
            tokens.dtype
        )
        context = torch.cat((tokens, condition[:, None]), 1)
        mask = torch.cat(
            (valid, torch.ones(b, 1, dtype=torch.bool, device=tokens.device)), 1
        )
        x = state.tokens
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                context,
                valid=mask,
                trace=trace,
                name="observe.attention" if i == 0 else f"observe.attention.{i}",
            )
        return x, self.scale(x).clamp(-5, 1)


class LatentDynamics(nn.Module):
    def __init__(self, width, action_width=2, depth=2):
        super().__init__()
        self.action_width = action_width
        self.action = nn.Linear(action_width + 1, width)
        self.uncertainty = nn.Linear(width, width)
        self.blocks = nn.ModuleList([Attend(width) for _ in range(depth)])
        self.delta, self.scale = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, state, action, dt, trace=None):
        x = state.tokens
        present = x.new_full((len(x), 1), float(action is not None))
        value = x.new_zeros(len(x), self.action_width) if action is None else action
        if (
            value.shape != (len(x), self.action_width)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Action must be finite [B,action_width]")
        condition = self.action(torch.cat((value, present), -1)) + position(
            dt, x.shape[-1]
        ).to(x.dtype)
        y = x + self.uncertainty(state.log_scale)
        for i, block in enumerate(self.blocks):
            y = block(
                y,
                torch.cat((y, condition[:, None]), 1),
                trace=trace,
                name=f"imagine.attention.{i}",
            )
        return x + self.delta(y) * dt[:, None, None].to(x.dtype), self.scale(y).clamp(
            -5, 1
        )


class Thinker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.read = Attend(width)
        self.feedback = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, query, context, feedback, trace=None, name="think"):
        if feedback.shape != (len(query), 3) or not torch.isfinite(feedback).all():
            raise ValueError("Self feedback must be finite [B,3]")
        context = torch.cat((context, self.feedback(feedback.detach())[:, None]), 1)
        return self.read(query, context, trace=trace, name=name + ".attention")


class ActionHead(nn.Module):
    def __init__(self, width, action_width=2):
        super().__init__()
        self.action_width = action_width
        self.output = nn.Linear(width, 2 * action_width)

    def forward(self, tokens):
        mean, log_scale = self.output(tokens.mean(1)).chunk(2, -1)
        # Distribution is Gaussian before tanh; outputs are normalized actions.
        return mean, log_scale.clamp(-5, 1)


class ErrorMonitor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, tokens):
        # The diagnostic head cannot move the world state to make error prediction easy.
        return F.softplus(self.output(tokens.detach().mean(1))).squeeze(-1)


class MultimodalAgent(nn.Module):
    def __init__(
        self,
        *,
        encoders,
        decoders,
        updater,
        dynamics,
        thinker,
        memory,
        action_head,
        monitor,
        feature_controller=None,
        task_interpreter=None,
        task_policy=None,
        metadata_encoder=None,
        width=32,
        groups=None,
    ):
        super().__init__()
        groups = groups or dict(
            sensory=16, entities=4, context=2, working=4, reasoning=4
        )
        required = {"sensory", "entities", "context", "working", "reasoning"}
        if set(groups) not in (required, {"world", "working", "reasoning"}) or any(
            v < 1 for v in groups.values()
        ):
            raise ValueError("Declare positive counts for all five latent roles")
        self.layout, offset = {}, 0
        for name, count in groups.items():
            self.layout[name] = slice(offset, offset + count)
            offset += count
        self.width, self.groups = width, dict(groups)
        self.initial = nn.Parameter(torch.randn(offset, width) * 0.02)
        self.encoders, self.decoders = nn.ModuleDict(encoders), nn.ModuleDict(decoders)
        self.updater, self.dynamics, self.thinker = updater, dynamics, thinker
        self.memory, self.action_head, self.monitor = memory, action_head, monitor
        self.feature_controller = feature_controller
        self.task_interpreter, self.task_policy = task_interpreter, task_policy
        self.metadata_encoder = metadata_encoder
        self.action_width = action_head.action_width
        if (
            updater.action_width != self.action_width
            or dynamics.action_width != self.action_width
        ):
            raise ValueError("Observation, action and dynamics widths must agree")

    def extra_repr(self):
        return f"width={self.width}, groups={self.groups}, action_width={self.action_width}"

    def initial_state(self, batch_size, time=0.0):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        tokens = self.initial[None].expand(batch_size, -1, -1).clone()
        now = batch_time(time, tokens)
        return LatentState(tokens, torch.zeros_like(tokens), now.clone(), now.clone())

    def forward(
        self,
        state,
        observations,
        *,
        time,
        previous_action=None,
        thinking_steps=1,
        feature_code=None,
        trace=None,
    ):
        """One observation/thinking cycle; memory writes and actions remain explicit."""
        state = self.observe(
            state,
            observations,
            time=time,
            previous_action=previous_action,
            feature_code=feature_code,
            trace=trace,
        )
        return self.think(state, steps=thinking_steps, trace=trace)

    def validate_state(self, state):
        if state.tokens.ndim != 3 or state.tokens.shape[1:] != self.initial.shape:
            raise ValueError("Latent token layout differs from this model")
        if state.log_scale.shape != state.tokens.shape:
            raise ValueError("Latent uncertainty shape differs from tokens")
        if (
            state.tokens.device != self.initial.device
            or state.tokens.dtype != self.initial.dtype
        ):
            raise ValueError("Latent state device/dtype differs from this model")
        if (
            state.time.shape != (len(state.tokens),)
            or state.observed_time.shape != state.time.shape
        ):
            raise ValueError("Latent times must be [B]")
        if any(
            not torch.isfinite(x).all()
            for x in (state.tokens, state.log_scale, state.time, state.observed_time)
        ):
            raise ValueError("Latent state must be finite")

    def propose_feature_code(self, state):
        self.validate_state(state)
        if self.feature_controller is None:
            raise ValueError("No feature controller supplied")
        context = torch.cat(
            [state.tokens[:, self.layout[name]] for name in ("working", "reasoning")], 1
        )
        return self.feature_controller(context)

    def encode(
        self,
        state,
        observations,
        *,
        time,
        feature_code=None,
        trace=None,
        purpose="observation",
    ):
        """Return independently inspectable input features under one frozen code.

        Code availability is state time for the controller, or the caller's cutoff
        for a user override. Ordinals within a pyramid refer to that input window's
        content only; cross-call state/code provenance is separately time-stamped.
        """
        self.validate_state(state)
        if purpose not in ("observation", "reflection"):
            raise ValueError("Unknown encoding purpose")
        now = batch_time(time, state.tokens)
        if (now < state.time).any():
            raise ValueError("Observation cannot move state backward in time")
        if not observations:
            raise ValueError(
                "Need at least one valid observation; use imagine for an unobserved interval"
            )
        unknown = set(observations) - set(self.encoders)
        if unknown:
            raise ValueError(f"No encoder supplied for modalities: {sorted(unknown)}")
        # Scan ALL raw inputs before ANY encoder is permitted to mix their values.
        for observation in observations.values():
            if purpose == "observation" and observation.provenance is not None:
                raise ValueError(
                    "Cannot assimilate generated content as an observation; use reflect"
                )
            if purpose == "reflection" and not isinstance(
                observation.provenance, Provenance
            ):
                raise ValueError("Reflection requires explicit generated provenance")
            _, times, valid = observation_values(observation)
            if len(times) != len(state.tokens):
                raise ValueError("Observation batch differs from latent interface")
            if ((times > now[:, None]) & valid).any():
                raise ValueError(
                    "Observation includes future information after the cutoff"
                )
        user_control = feature_code is not None
        if not user_control and self.feature_controller is not None:
            feature_code = self.propose_feature_code(state)
        condition_time = now if user_control else state.time
        if feature_code is not None and trace is not None:
            trace["encode.feature_code"] = feature_code.detach().cpu().clone()
            trace["encode.condition_time"] = condition_time.detach().cpu().clone()
            trace["encode.condition_source"] = (
                "user" if user_control else "pre_observation_state"
            )
        encoded = {}
        for name, encoder in self.encoders.items():
            if name not in observations:
                continue
            if hasattr(encoder, "code_width"):
                local_trace = {} if trace is not None else None
                encoded[name] = encoder(
                    observations[name],
                    condition=feature_code,
                    condition_time=condition_time,
                    trace=local_trace,
                )
                if trace is not None:
                    trace.update(
                        {
                            f"encode.{name}.{key}": value
                            for key, value in local_trace.items()
                        }
                    )
            else:
                # Existing one-scale custom adapters keep their simple contract.
                encoded[name] = encoder(observations[name])
        return encoded

    def observe(
        self,
        state,
        observations,
        *,
        time,
        previous_action=None,
        feature_code=None,
        trace=None,
    ):
        encoded = self.encode(
            state, observations, time=time, feature_code=feature_code, trace=trace
        )
        now = batch_time(time, state.tokens)
        batches, labels = [], []
        scale_labels = []
        for name, features in encoded.items():
            if isinstance(features, FeaturePyramid):
                item = features.as_tokens()
                scale_labels.extend(
                    f"{name}.scale.{i}"
                    for i, scale in enumerate(features.scales)
                    for _ in range(scale.values.shape[1])
                )
            else:
                item = features
                scale_labels.extend([name] * item.values.shape[1])
            if (
                item.values.shape[0] != len(state.tokens)
                or item.values.shape[-1] != self.width
            ):
                raise ValueError("Encoder batch/width differs from latent interface")
            if ((item.times > now[:, None]) & item.valid).any():
                raise ValueError(
                    "Observation includes future information after the cutoff"
                )
            values = item.values + position(item.times - now[:, None], self.width).to(
                item.values.dtype
            )
            values = values.masked_fill(~item.valid[:, :, None], 0)
            batches.append((values, item.valid))
            labels.extend([name] * values.shape[1])
        values, valid = (torch.cat([item[i] for item in batches], 1) for i in range(2))
        if not valid.any(1).all():
            raise ValueError("Each sample needs at least one valid observation")
        tokens, scale = self.updater(state, values, valid, previous_action, now, trace)
        result = replace(
            state,
            tokens=tokens,
            log_scale=scale,
            time=now.clone(),
            observed_time=now.clone(),
            imagined=False,
            thinking_steps=0,
            observation_count=state.observation_count + 1,
        )
        if trace is not None:
            trace["observe.input_scales"] = scale_labels + [
                "previous_action_and_elapsed_time"
            ]
            trace["observe.input_modalities"] = labels + [
                "previous_action_and_elapsed_time"
            ]
            trace["observe.valid"] = valid.detach().cpu().clone()
        record_state(trace, "observe", result)
        return result

    def remember(self, state, *, source):
        self.validate_state(state)
        return self.memory.write(state, source)

    def reflect(self, state, observations, *, trace=None):
        """Read tagged generated content into working/reasoning only.

        Keep a clean observational branch for future memory writes. Ancestry on
        this branch is persistent, even if real observations subsequently arrive.
        """
        if self.metadata_encoder is None:
            raise ValueError("Reflection requires a metadata encoder")
        for name, observation in observations.items():
            p = observation.provenance
            if (
                isinstance(p, Provenance)
                and p.origin == "generated"
                and p.request.modality != name
            ):
                raise ValueError("Generated modality differs from its provenance")
        input_trace = {} if trace is not None else None
        encoded = self.encode(
            state,
            observations,
            time=state.time,
            trace=input_trace,
            purpose="reflection",
        )
        if trace is not None:
            trace.update({f"reflect.{k}": v for k, v in input_trace.items()})
        context, provenance = [], list(state.generated_ancestry)
        for name, features in encoded.items():
            item = (
                features.as_tokens()
                if isinstance(features, FeaturePyramid)
                else features
            )
            # Attend's context has no padding mask in think; pool valid features
            # into one token per modality rather than introducing invalid keys.
            pooled = (item.values * item.valid[..., None]).sum(1) / item.valid.sum(
                1
            ).clamp_min(1)[:, None]
            if not item.valid.any(1).all():
                raise ValueError("Reflection requires valid generated features")
            p = observations[name].provenance
            metadata = self.metadata_encoder([p.to_dict()] * len(state.tokens))
            if trace is not None:
                trace[f"reflect.{name}.metadata"] = metadata.detach().cpu().clone()
            context.extend((pooled[:, None], metadata))
            if p not in provenance:
                provenance.append(p)
        result = self.think(state, goal=torch.cat(context, 1), trace=trace)
        result = replace(result, generated_ancestry=tuple(provenance))
        if trace is not None:
            trace["reflect.provenance"] = [p.to_dict() for p in provenance]
        record_state(trace, "reflect", result)
        return result

    def task_tokens(self, state, sessions, *, trace=None):
        self.validate_state(state)
        if self.task_interpreter is None or self.metadata_encoder is None:
            raise ValueError("No task interpreter/metadata encoder supplied")
        if len(sessions) != len(state.tokens):
            raise ValueError("One task session is required per batch member")
        ids, valid = bytes_batch(
            [s.request.instruction for s in sessions], state.tokens.device
        )
        observation = Observation(ids, state.time[:, None].expand_as(ids), valid)
        input_trace = {} if trace is not None else None
        features = self.encode(
            state, {"text": observation}, time=state.time, trace=input_trace
        )["text"]
        if trace is not None:
            trace.update({f"task.{k}": v for k, v in input_trace.items()})
        instruction = (
            features.as_tokens() if isinstance(features, FeaturePyramid) else features
        )
        records = [s.context_record() for s in sessions]
        # Instruction text is read through the shared modality encoder only.
        for record in records:
            record["request"].pop("instruction")
        metadata = self.metadata_encoder(records)
        tokens = self.task_interpreter(state.tokens, instruction, metadata, trace=trace)
        if trace is not None:
            trace["task.tokens"] = tokens.detach().cpu().clone()
            trace["task.metadata"] = metadata.detach().cpu().clone()
            trace["task.requests"] = [s.request.to_dict() for s in sessions]
        return tokens

    def task_predictions(self, state, sessions, *, trace=None):
        if self.task_policy is None:
            raise ValueError("No task policy supplied")
        prediction = self.task_policy(self.task_tokens(state, sessions, trace=trace))
        if trace is not None:
            for name in ("operation_logits", "modality_logits", "completion_logits"):
                trace["task." + name] = getattr(prediction, name).detach().cpu().clone()
            trace["task.modalities"] = list(prediction.modalities)
        return prediction

    def decide(self, state, session, *, agent, trace=None):
        if len(state.tokens) != 1:
            raise ValueError(
                "Interactive tasks use one stream; task_predictions supports batches"
            )
        return self.task_predictions(state, [session], trace=trace).select(
            session, agent
        )

    def emit(
        self,
        state,
        session,
        requests,
        *,
        produced_by,
        video_states=None,
        max_text_tokens=32,
        trace=None,
    ):
        """Validate every output first, then return successful outputs and errors.

        A decoder failure stops this batch and retains earlier successes. Only
        successful answers fulfill requirements. Disabled controls apply to the
        output surface; video may still use the internal image decoder.
        """
        self.validate_state(state)
        if len(state.tokens) != 1 or session.finished or session.failure is not None:
            raise ValueError("Emission needs one active task stream")
        if produced_by.role != "agent":
            raise ValueError("This model's producer must be an explicit agent")
        requests = tuple(requests)
        controls = {c.modality: c for c in session.request.controls}
        available = set(self.decoders) | (
            {"video"} if "image" in self.decoders else set()
        )
        if set(controls) - available:
            raise ValueError("Controlled modality has no output adapter")
        seen = {o.provenance.request.output_id for o in session.outputs}
        for request in requests:
            if request.output_id in seen:
                raise ValueError("Cannot emit duplicate output IDs")
            seen.add(request.output_id)
            if request.in_response_to != session.request.task_id:
                raise ValueError("Output request belongs to a different task")
            if request.modality not in available:
                raise ValueError("Requested modality has no output adapter")
            control = controls.get(request.modality)
            if control and control.mode == "disabled":
                raise ValueError("Requested output is disabled")
            if (
                control
                and control.mode == "required"
                and request.purpose == "answer"
                and request.requested_by != control.specified_by
            ):
                raise ValueError(
                    "Required output requester must match the control author"
                )
            if (
                control is None or control.mode == "automatic"
            ) and request.requested_by.role != "agent":
                raise ValueError("Automatic output requester must be an agent")
            if request.modality == "text" and (
                not isinstance(max_text_tokens, int) or max_text_tokens < 1
            ):
                raise ValueError("Text generation budget must be positive")
            if request.modality == "video":
                if video_states is None or not len(video_states):
                    raise ValueError("Video emission requires an explicit trajectory")
                for frame in video_states:
                    self.validate_state(frame)
                    if len(frame.tokens) != 1:
                        raise ValueError("Video trajectory batch differs from task")
                for a, b in zip(video_states, video_states[1:]):
                    if (b.time <= a.time).any():
                        raise ValueError("Video trajectory must advance time")
        outputs, errors = [], {}
        if requests:
            goal = self.task_tokens(state, [session], trace=trace)
            conditioned = self.think(state, goal=goal, trace=trace)
        for request in requests:
            try:
                if request.modality == "text":
                    values = self.generate_text(conditioned, max_tokens=max_text_tokens)
                elif request.modality == "video":
                    frames = [
                        self.think(s, goal=goal, trace=trace) for s in video_states
                    ]
                    values = self.decode_video(frames, trace=trace)
                else:
                    values = self.decoders[request.modality](
                        conditioned.tokens, trace=trace
                    )
                if (
                    not isinstance(values, torch.Tensor)
                    or values.ndim < 2
                    or values.numel() == 0
                    or len(values) != 1
                    or not torch.isfinite(values).all()
                ):
                    raise ValueError(
                        "Output adapter must produce a finite tensor with batch size one"
                    )
                dependencies = list(state.generated_ancestry) + [
                    o.provenance for o in session.outputs
                ]
                if request.modality == "video":
                    dependencies.extend(
                        p for s in video_states for p in s.generated_ancestry
                    )
                ancestors = tuple(
                    dict.fromkeys(
                        (
                            *request.parents,
                            *(
                                identifier
                                for p in dependencies
                                for identifier in (*p.ancestors, p.request.output_id)
                            ),
                        )
                    )
                )
                outputs.append(
                    GeneratedOutput(
                        values, Provenance(request, produced_by, ancestors=ancestors)
                    )
                )
            except Exception as exc:
                errors[request.output_id] = f"{type(exc).__name__}: {exc}"
                break
        updated = replace(session, outputs=session.outputs + tuple(outputs))
        if trace is not None:
            trace["emit.outputs"] = [o.provenance.to_dict() for o in outputs]
            trace["emit.errors"] = errors.copy()
            trace["emit.remaining"] = list(updated.remaining)
        return Emission(updated, tuple(outputs), errors)

    def step_task(
        self,
        state,
        session,
        *,
        agent,
        dt=1.0,
        video_states=None,
        max_text_tokens=32,
        trace=None,
    ):
        """One bounded learned proposal/execution; physical actions are returned only.

        Ask is surfaced to the caller as an operation. It does not silently enable
        text. The byte decoder is not a trained general chatbot.
        Imagination returns a separate branch and leaves the live clock unchanged.
        """
        selection = self.decide(state, session, agent=agent, trace=trace)
        operation = selection.operation
        if trace is not None:
            trace["task.selection"] = {
                "operation": operation,
                "raw_operation": selection.raw_operation,
                "raw_modalities": list(selection.raw_modalities),
            }
        if operation == "finish":
            return TaskStep(state, replace(session, finished=True), selection)
        if operation == "failed":
            return TaskStep(state, session, selection)
        if operation == "emit":
            result = self.emit(
                state,
                session,
                selection.requests,
                produced_by=agent,
                video_states=video_states,
                max_text_tokens=max_text_tokens,
                trace=trace,
            )
            return TaskStep(state, result.session, selection, emission=result)
        if operation == "ask":
            # The caller can request clarification in an enabled modality. Do not
            # silently enable text or fabricate a user request for it.
            return TaskStep(state, session, selection)
        goal = self.task_tokens(state, [session], trace=trace)
        ancestry = tuple(
            dict.fromkeys(
                (*state.generated_ancestry, *(o.provenance for o in session.outputs))
            )
        )
        conditioned = replace(
            self.think(state, goal=goal, trace=trace), generated_ancestry=ancestry
        )
        if operation == "recall":
            retrieved = self.memory.read(conditioned, trace=trace, name="task.recall")
            if retrieved.shape[1]:
                conditioned = self.think(
                    conditioned, goal=torch.cat((goal, retrieved), 1), trace=trace
                )
        if operation == "imagine":
            return TaskStep(
                state,
                session,
                selection,
                imagined=self.imagine(conditioned, dt=dt, trace=trace),
            )
        if operation == "act":
            return TaskStep(
                conditioned, session, selection, action=self.propose_action(conditioned)
            )
        return TaskStep(conditioned, session, selection)

    def think(self, state, *, steps=1, goal=None, feedback=None, trace=None):
        self.validate_state(state)
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Thinking steps must be a nonnegative integer")
        indices = torch.cat(
            [
                torch.arange(
                    self.layout[name].start,
                    self.layout[name].stop,
                    device=state.tokens.device,
                )
                for name in ("working", "reasoning")
            ]
        )
        if goal is not None and (
            goal.ndim != 3
            or len(goal) != len(state.tokens)
            or goal.shape[-1] != self.width
        ):
            raise ValueError("Goal must be latent tokens [B,G,D]")
        for i in range(steps):
            name = f"think.{i}"
            recalled = self.memory.read(state, trace=trace, name=name + ".memory")
            context = torch.cat((state.tokens, recalled), 1)
            if goal is not None:
                context = torch.cat((context, goal), 1)
            diagnostic = feedback
            if diagnostic is None:
                diagnostic = torch.stack(
                    (
                        self.monitor(state.tokens),
                        state.log_scale.exp().mean((1, 2)),
                        state.tokens.new_full(state.time.shape, state.thinking_steps),
                    ),
                    -1,
                )
            update = self.thinker(
                state.tokens[:, indices], context, diagnostic, trace=trace, name=name
            )
            tokens = state.tokens.clone()
            tokens[:, indices] = update
            state = replace(
                state, tokens=tokens, thinking_steps=state.thinking_steps + 1
            )
            record_state(trace, name, state)
        return state

    def imagine(self, state, action=None, *, dt=1.0, sample=False, trace=None):
        self.validate_state(state)
        interval = batch_time(dt, state.tokens)
        if (interval <= 0).any():
            raise ValueError("Imagination time interval must be positive")
        mean, scale = self.dynamics(state, action, interval, trace)
        tokens = mean + torch.randn_like(mean) * scale.exp() if sample else mean
        result = replace(
            state,
            tokens=tokens,
            log_scale=scale,
            time=state.time + interval,
            imagined=True,
            thinking_steps=0,
        )
        if trace is not None:
            trace["imagine.mean"] = mean.detach().cpu().clone()
            trace["imagine.action"] = (
                None if action is None else action.detach().cpu().clone()
            )
        record_state(trace, "imagine", result)
        return result

    def decode(self, state, *, text_prefix=None, modalities=None, trace=None):
        self.validate_state(state)
        names = list(self.decoders) if modalities is None else list(modalities)
        if set(names) - set(self.decoders):
            raise ValueError("Requested output has no decoder")
        output = {}
        for name in names:
            if name == "text":
                if text_prefix is not None:
                    output[name] = self.decoders[name](
                        state.tokens, text_prefix, trace=trace
                    )
            else:
                output[name] = self.decoders[name](state.tokens, trace=trace)
        return output

    def decode_video(self, states, trace=None):
        if not states:
            raise ValueError("Video needs a nonempty latent trajectory")
        for a, b in zip(states, states[1:]):
            if (b.time <= a.time).any():
                raise ValueError("Video states must advance time")
        return torch.stack(
            [self.decoders["image"](s.tokens, trace=trace) for s in states], 1
        )

    def generate_text(self, state, *, max_tokens=32):
        self.validate_state(state)
        return self.decoders["text"].generate(state.tokens, max_tokens=max_tokens)

    def propose_action(self, state, *, sample=False):
        self.validate_state(state)
        mean, scale = self.action_head(state.tokens)
        raw = mean + scale.exp() * torch.randn_like(mean) if sample else mean
        return raw.tanh()

    def intervene(self, state, role, value):
        """Replace one group's activations without mutating the caller's state."""
        if role not in self.layout:
            raise ValueError(f"Unknown latent role {role!r}")
        self.validate_state(state)
        tokens = state.tokens.clone()
        tokens[:, self.layout[role]] = value
        return replace(state, tokens=tokens)
