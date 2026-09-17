import torch
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pathwm.evaluation.request_meaning import continuation_scores
from pathwm.models.modalities import TextDecoder


def test_continuation_scores_use_last_byte_without_teacher_forced_eos():
    torch.manual_seed(18)
    decoder = TextDecoder(16).train()
    tokens = torch.randn(3, 4, 16)
    texts = ["rot", "grün", ""]
    before = torch.get_rng_state().clone()
    rows = continuation_scores(decoder, tokens, texts)
    assert decoder.training and torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        for i, (text, row) in enumerate(zip(texts, rows)):
            prefix = torch.tensor([[1, *(b + 3 for b in text.encode())]])
            # Eval mode can select different kernels, so compare with the same mode.
            decoder.eval()
            logits = decoder(tokens[i : i + 1], prefix)[0, -1]
            decoder.train()
            p = logits.softmax(-1)
            assert row["prefix_tokens"] == prefix.shape[1]
            torch.testing.assert_close(torch.tensor(row["eos_probability"]), p[2])
            torch.testing.assert_close(torch.tensor(row["space_probability"]), p[35])
            torch.testing.assert_close(
                torch.tensor(row["space_minus_eos_logit"]), logits[35] - logits[2]
            )
            assert row["top_legal_id"] == int(logits[2:].argmax()) + 2


def test_continuation_scores_keep_full_vocab_mass_separate_from_legal_argmax():
    decoder = TextDecoder(16)
    with torch.no_grad():
        decoder.output.weight.zero_()
        decoder.output.bias.zero_()
        decoder.output.bias[0] = 100
        decoder.output.bias[2] = 10
    row = continuation_scores(decoder, torch.zeros(1, 2, 16), ["rot"])[0]
    assert row["top_legal_id"] == 2
    assert row["eos_probability"] < 1e-30


def test_cached_completion_reproduces_answers_without_running_core(
    tmp_path, monkeypatch
):
    from experiments import modality_readout as recipe
    from pathwm.io import atomic_json, file_hash
    from pathwm.models.modalities import bytes_text

    torch.manual_seed(419)
    model = recipe.Model("native").eval()
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    (source / "last.pt").write_bytes(b"fixed source fixture")
    source_hash = file_hash(source / "last.pt")
    monkeypatch.setattr(recipe, "restore_readout", lambda *a: (model, {}, source_hash))

    def no_core(*a, **kw):
        raise AssertionError("Cached diagnosis must not resample the core")

    monkeypatch.setattr(model.core, "forward", no_core)
    tokens = torch.randn(2, 4, model.outputs.decoders["text"].width)
    with torch.no_grad():
        ids = model.outputs.generate(tokens, 16)
    rows = [
        dict(
            condition="full",
            format="first",
            wording="test/0",
            draw=0,
            clip=str(i),
            pair="pair",
            expected=expected,
            generated=bytes_text(generated),
            ended=bool((generated == 2).any()),
            first_word=False,
            exact=False,
        )
        for i, (expected, generated) in enumerate(zip(["rot", "grün"], ids))
    ]
    atomic_json(
        cache / "run.json",
        dict(
            identity=dict(
                settings=dict(source_sha256=source_hash, observation_question="full")
            )
        ),
    )
    atomic_json(cache / "status.json", dict(result="complete"))
    atomic_json(cache / "request_meanings.json", dict(examples=rows))
    np.savez(cache / "request_states.npz", working=tokens.numpy())
    torch.save(dict(model=model.state_dict()), cache / "last.pt")
    args = SimpleNamespace(
        core=source,
        request_cache=cache,
        output=tmp_path / "output",
        seed=4,
        device="cpu",
    )
    recipe.request_completion(args)
    result = json.loads((args.output / "continuations.json").read_text())["examples"]
    assert len(result) == len(rows)
    assert result[0]["continuation"]["expected_first"]["prefix"] == "rot"
    assert (args.output / "report.html").exists()
    assert json.loads((args.output / "status.json").read_text())["result"] == "complete"
    # Misaligned/mismatched caches must fail before opening an output run.
    args.output = tmp_path / "bad-source"
    (source / "last.pt").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="disagree"):
        recipe.request_completion(args)
    assert not args.output.exists()
    (source / "last.pt").write_bytes(b"fixed source fixture")
    np.savez(cache / "request_states.npz", working=tokens[:1].numpy())
    with pytest.raises(ValueError, match="aligned"):
        recipe.request_completion(args)
    assert not args.output.exists()
