import copy

import numpy as np
import torch
from torch.nn import functional as F

from experiments import modality_readout as recipe


def test_retention_kl_includes_first_eos_but_not_padding_or_continuation():
    torch.manual_seed(4)
    student = torch.randn(2, 5, 7, requires_grad=True)
    teacher = torch.randn_like(student)
    labels = torch.tensor([[3, 2, 2, 0, 4], [4, 5, 6, 2, 0]])
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    loss = recipe.retention_kl(student, teacher, labels)
    q = teacher.softmax(-1)
    reference = q * (teacher.log_softmax(-1) - student.log_softmax(-1))
    reference = reference.sum(-1)[mask].mean() / np.log(7)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert torch.count_nonzero(student.grad[~mask]) == 0
    assert student.grad[mask].abs().sum() > 0


def test_retention_teacher_preserves_rng_and_identity_then_only_student_learns():
    torch.set_num_threads(2)
    recipe.seed_everything(94)
    model = recipe.Model("native")
    recipe.configure_grounded_training(model, "core")
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    teacher_hash = recipe.state_hash(teacher)
    population = recipe.dataset("train", 7201)
    indices = [0, 2]
    inputs = recipe.observations(population, "all", indices, "cpu")
    targets = recipe.target_batch(population, indices, "cpu")
    normalizers = recipe.scales(population)
    before = torch.get_rng_state().clone()
    reference = recipe.retention_targets(teacher, inputs, targets)
    assert torch.equal(torch.get_rng_state(), before)
    tokens = model.core(inputs)
    loss, terms = recipe.retention_objective(
        model, tokens, targets, reference, normalizers
    )
    assert abs(float(loss.detach())) < 1e-6
    assert set(terms) == {"text_target", "text_greedy", "image", "audio", "video"}
    assert all(not v.requires_grad for v in reference.values())
    # Perturb the shared state, so every output can supply a corrective gradient.
    changed = (tokens.detach() + 0.1 * torch.randn_like(tokens)).requires_grad_()
    loss, terms = recipe.retention_objective(
        model, changed, targets, reference, normalizers
    )
    assert loss > 0
    loss.backward()
    assert changed.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.outputs.decoders["text"].parameters()
    )
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in model.core.agent.encoders.parameters())
    assert recipe.state_hash(teacher) == teacher_hash
    # Numeric modality reference; the average prefix terms must not double text.
    expected = 0.5 * (terms["text_target"] + terms["text_greedy"])
    for kind in ("image", "audio", "video"):
        numeric = F.mse_loss(model.outputs(kind, changed), reference[kind])
        numeric = numeric / normalizers[kind]
        assert np.isclose(terms[kind], float(numeric.detach()))
        expected += terms[kind]
    assert np.isclose(float(loss.detach()), expected)
