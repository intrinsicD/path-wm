import torch
from world_model.curriculum.encoder_variants import ExperimentalEncoder
from world_model.curriculum.perception_extensions import ContinuationEncoder, encoder_optimizer_groups


def test_added_blocks_initially_preserve_source_function_and_then_receive_gradients():
    torch.set_num_threads(1); torch.manual_seed(51)
    base = ExperimentalEncoder(depth=2, exchange=True).eval()
    rgb = torch.rand(2, 3, 64, 64)
    reference = base(rgb).tokens().detach()
    for kind in ('joint', 'conv', 'transformer'):
        encoder = ContinuationEncoder(kind, 9107, base.state_dict()).eval()
        actual = encoder(rgb).tokens()
        assert torch.equal(reference, actual.detach()), kind
        if kind == 'joint': continue
        actual.square().mean().backward()
        gates = [p for n, p in encoder.named_parameters() if '.gate' in n]
        assert any(p.grad.abs().sum() > 0 for p in gates)
        branches = [p for n, p in encoder.named_parameters() if n.startswith('extensions.') and '.gate' not in n]
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in branches)
        with torch.no_grad():
            for gate in gates: gate.fill_(.1)
        encoder.zero_grad(); encoder(rgb).tokens().square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branches)


def test_gate_and_normalization_decay_and_learning_rates_are_explicit():
    torch.manual_seed(32)
    base = ExperimentalEncoder(depth=2, exchange=True)
    encoder = ContinuationEncoder('transformer', 9107, base.state_dict())
    groups = encoder_optimizer_groups(encoder)
    by_id = {id(p): group for group in groups for p in group['params']}
    assert len(by_id) == sum(1 for _ in encoder.parameters())
    for name, parameter in encoder.named_parameters():
        assert by_id[id(parameter)]['lr'] == (3e-4 if name.startswith('extensions.') else 3e-5)
        if '.gate' in name: assert by_id[id(parameter)]['weight_decay'] == 0
    for module in encoder.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            assert all(by_id[id(p)]['weight_decay'] == 0 for p in module.parameters(recurse=False))
