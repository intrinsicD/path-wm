"""Opt-in shape accounting, deliberately separate from performance timing.

No attention matrices, device transfers, tensor copies or hooks on backward. Masks
do not shrink the allocated lengths reported here. FLOPs cover attention matmuls
and projections only, excluding MLPs, encoders, normalization and memory traffic.
"""

from collections import defaultdict

from torch import nn

from pathwm.models.blocks import Attention
from pathwm.models.modalities import Attend
from pathwm.models.multiscale import ConditionedBlock, FeaturePyramid


class Workload:
    def __init__(self, model):
        self.model = model
        self.attention, self.encoded, self.observation, self.states = [], [], [], []
        self.handles = []

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Workload context is already active")
        covered = set()
        for name, module in self.model.named_modules():
            if isinstance(module, (Attend, ConditionedBlock, Attention)):
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._attention_hook(name), with_kwargs=True
                    )
                )
                covered.update(id(m) for m in module.modules())
            elif (
                isinstance(module, nn.MultiheadAttention) and id(module) not in covered
            ):
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._attention_hook(name), with_kwargs=True
                    )
                )
        encoders = getattr(self.model, "encoders", {})
        for name, encoder in encoders.items():
            self.handles.append(encoder.register_forward_hook(self._encoder_hook(name)))
        updater = getattr(self.model, "updater", None)
        if updater is not None:
            self.handles.append(
                updater.register_forward_pre_hook(self._observation_hook)
            )
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _attention_hook(self, name):
        def hook(module, args, kwargs):
            query = args[0] if args else kwargs["query"]
            if isinstance(module, ConditionedBlock):
                context = kwargs.get("context")
                kind = "self" if context is None else "cross"
                key = query if context is None else context
                query, key = query.values, key.values
                heads, width = module.heads, module.attention.embed_dim
            else:
                key = (
                    args[1]
                    if len(args) > 1
                    else kwargs["context" if isinstance(module, Attend) else "key"]
                )
                kind = "self" if query is key else "cross"
                attention = module.attention if isinstance(module, Attend) else module
                heads = (
                    attention.heads
                    if isinstance(attention, Attention)
                    else attention.num_heads
                )
                width = (
                    attention.width
                    if isinstance(attention, Attention)
                    else attention.embed_dim
                )
            if isinstance(module, nn.MultiheadAttention) and not module.batch_first:
                batch, nq, nk = query.shape[1], query.shape[0], key.shape[0]
            else:
                batch, nq, nk = query.shape[0], query.shape[1], key.shape[1]
            scope = name.split(".")[0] or "root"
            self.attention.append(
                dict(
                    layer=name or "root",
                    scope=scope,
                    kind=kind,
                    batch=batch,
                    N_query=nq,
                    N_key=nk,
                    heads=heads,
                    width=width,
                    score_elements=batch * heads * nq * nk,
                    # Multiply and add count separately. V has the same width as K
                    # for the repository's supported modules.
                    attention_matmul_flops=4 * batch * nq * nk * width,
                    projection_flops=2
                    * batch
                    * width
                    * (nq * query.shape[-1] + 2 * nk * key.shape[-1] + nq * width),
                )
            )

        return hook

    def _encoder_hook(self, name):
        def hook(module, args, output):
            scales = output.scales if isinstance(output, FeaturePyramid) else (output,)
            sizes = [s.values.shape[1] for s in scales]
            self.encoded.append(dict(modality=name, N_encoded=sum(sizes), scales=sizes))

        return hook

    def _observation_hook(self, module, args):
        self.observation.append(dict(N_observation=args[1].shape[1]))

    def record_state(self, state):
        self.states.append(
            dict(
                N_world_state=state.h.shape[1],
                N_persistent_total=state.tokens.shape[1],
                N_evidence=state.evidence.shape[1],
                thinking_steps=state.thinking_steps,
            )
        )

    def summary(self):
        layers = defaultdict(lambda: dict(calls=0, N_query=[], N_key=[], kinds=[]))
        scopes = defaultdict(
            lambda: dict(
                query_tokens=0,
                self_pairs=0,
                cross_pairs=0,
                score_elements=0,
                attention_flops=0,
                batch_query_tokens=0,
                batch_self_pairs=0,
                batch_cross_pairs=0,
            )
        )
        for row in self.attention:
            layer = layers[row["layer"]]
            layer["calls"] += 1
            for key in ("N_query", "N_key"):
                layer[key].append(row[key])
            layer["kinds"].append(row["kind"])
            scope = scopes[row["scope"]]
            scope["query_tokens"] += row["N_query"]
            scope["batch_query_tokens"] += row["batch"] * row["N_query"]
            scope["batch_" + row["kind"] + "_pairs"] += (
                row["batch"] * row["N_query"] * row["N_key"]
            )
            scope[row["kind"] + "_pairs"] += row["N_query"] * row["N_key"]
            scope["score_elements"] += row["score_elements"]
            scope["attention_flops"] += (
                row["attention_matmul_flops"] + row["projection_flops"]
            )
        return dict(
            attention=self.attention,
            encoded=self.encoded,
            observation=self.observation,
            states=self.states,
            per_layer=dict(layers),
            per_scope=dict(scopes),
            query_token_evaluations=sum(r["query_tokens"] for r in scopes.values()),
            self_attention_pairs=sum(r["self_pairs"] for r in scopes.values()),
            cross_attention_pairs=sum(r["cross_pairs"] for r in scopes.values()),
            batch_query_token_evaluations=sum(
                r["batch_query_tokens"] for r in scopes.values()
            ),
            batch_self_attention_pairs=sum(
                r["batch_self_pairs"] for r in scopes.values()
            ),
            batch_cross_attention_pairs=sum(
                r["batch_cross_pairs"] for r in scopes.values()
            ),
            limits="Allocated shapes, not sparse useful pairs or measured FLOPs. "
            "Layer calls include repeated corrections as well as loops. "
            "Attention FLOPs exclude MLP/conv/norm/masks/transfers.",
        )
