from dataclasses import replace

import pytest
import torch

from experiments.world_state import FoundationModel
from pathwm.models.belief_state import Packet
from pathwm.models.modalities import (
    AudioDecoder, ImageDecoder, TextDecoder, Observation, TokenBatch, bytes_batch,
)
from pathwm.models.tasks import Actor, OutputRequest, Provenance
from pathwm.world_state.modules import Candidate, CandidateEncoder
from pathwm.world_state.retrieval import Query


def observation(kind):
    if kind == "text":
        values, valid = bytes_batch(["Tür öffnen"])
        return Observation(values, torch.ones_like(values, dtype=torch.float64), valid)
    values = (torch.randn(1, 2, 32) if kind == "audio"
              else torch.rand(1, 2, 3, 16, 16))
    return Observation(values, torch.tensor([[0.5, 1.0]], dtype=torch.float64))


@pytest.mark.parametrize("kind", ["text", "audio", "video", "image"])
def test_foundation_each_source_reaches_store_reasoner_and_outputs(kind):
    torch.manual_seed(33)
    model = FoundationModel().eval()
    obs = observation(kind)
    encoded = model.agent.encoders[kind](obs)
    projection = CandidateEncoder(16, 4, 4)
    keys, values = projection.pool(encoded)
    candidate = Candidate("window", kind, kind, keys[0, 0], values[0, 0], kind, "v1")
    session = model.session()
    response = session.observe(kind, occurred_at=1, available_at=1,
                               candidates=(candidate,), packets=(Packet(kind, kind, obs),))
    assert response["bindings"][0]["status"] == "new"
    assert session.state.observation_count == 1
    snapshot = session.snapshot()
    restored = model.session(snapshot)
    query = Query(entity_ids=(response["bindings"][0]["entity_id"],))
    state, context, _ = session.think(query)
    replay, _, _ = restored.think(query)
    assert torch.equal(state.tokens, replay.tokens)
    assert session.store.snapshot() == snapshot["world"]
    assert context.components and session.store.evidence()[0].modality == kind
    output = model.agent.decode(state, text_prefix=torch.ones(1, 1, dtype=torch.long))
    assert set(output) == {"image", "audio", "text"}
    assert all(torch.isfinite(x).all() for x in output.values())
    assert model.agent.generate_text(state, max_tokens=3).shape[1] <= 4
    future = model.agent.imagine(state, None, dt=1)
    assert model.agent.decode_video([state, future]).shape == (1, 2, 3, 16, 16)


def test_candidate_pool_excludes_invalid_and_unselected_values_and_gradients():
    projection = CandidateEncoder(4, 4, 2)
    x = torch.randn(2, 5, 4, requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False]]).expand(2, -1)
    selection = torch.zeros(2, 1, 5, dtype=torch.bool)
    selection[:, :, 0] = True
    batch = TokenBatch(x.masked_fill(~valid[..., None], float("nan")), torch.zeros(2, 5), valid)
    keys, values = projection.pool(batch, selection)
    assert keys.shape == (2, 1, 4) and values.shape == (2, 1, 2)
    values.sum().backward()
    assert x.grad[:, 0].abs().sum() > 0
    assert x.grad[:, 1:].count_nonzero() == 0
    with pytest.raises(ValueError, match="valid"):
        projection.pool(batch, torch.zeros_like(selection))


@pytest.mark.parametrize("kind", ["text", "audio", "image"])
def test_output_context_masks_exclude_nans_and_gradients(kind):
    torch.manual_seed(4)
    module = {"text": TextDecoder, "audio": AudioDecoder, "image": ImageDecoder}[kind](16)
    tokens = torch.randn(2, 3, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False]]).expand(2, -1)
    changed = tokens.masked_fill(~valid[..., None], float("nan"))
    args = (torch.ones(2, 1, dtype=torch.long),) if kind == "text" else ()
    expected = module(tokens[:, :2], *args)
    output = module(changed, *args, valid=valid)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert tokens.grad[:, :2].abs().sum() > 0
    assert tokens.grad[:, 2].count_nonzero() == 0


def test_generated_candidate_cannot_be_written_as_source():
    model = FoundationModel().eval()
    provenance = Provenance(OutputRequest("out", "text", Actor("user", "user"), "request"), Actor("agent", "model"))
    c = Candidate("c", "decoder", "text", torch.ones(4), torch.ones(4), "text", "1",
                  provenance=provenance)
    session = model.session()
    before = session.snapshot()
    with pytest.raises(ValueError, match="source"):
        session.observe("bad", occurred_at=1, available_at=1, candidates=(c,))
    assert session.store.snapshot() == before["world"]


def test_video_decode_rejects_invalid_state():
    model = FoundationModel().agent.eval()
    state = model.initial_state(1)
    bad = replace(state, time=torch.full_like(state.time, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        model.decode_video([bad])
