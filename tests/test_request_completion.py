import torch

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
            logits = decoder(tokens[i:i+1], prefix)[0, -1]
            decoder.train()
            p = logits.softmax(-1)
            assert row["prefix_tokens"] == prefix.shape[1]
            torch.testing.assert_close(torch.tensor(row["eos_probability"]), p[2])
            torch.testing.assert_close(torch.tensor(row["space_probability"]), p[35])
            torch.testing.assert_close(torch.tensor(row["space_minus_eos_logit"]), logits[35] - logits[2])
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
