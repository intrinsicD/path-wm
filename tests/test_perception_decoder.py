"""Decoder input evidence and conditioning must be identifiable and aligned."""
import torch
from world_model.curriculum.perception_decoder import pack_native, raw_patches, DenseDecoder


def test_native_normalization_preserves_channel_mean_and_scale():
    torch.manual_seed(87)
    tokens = torch.randn(2, 7, 48) * 3 + 4
    tokens[0, 0] = 6
    packed = pack_native(tokens)
    restored = packed[..., :-2] * packed[..., -1:] + packed[..., -2:-1]
    assert torch.allclose(tokens, restored, atol=1e-6)


def test_raw_patch_tokens_reconstruct_the_same_source_pixels():
    rgb = torch.arange(3 * 64 * 64).reshape(1, 3, 64, 64).float()
    tokens = raw_patches(rgb)
    restored = torch.nn.functional.pixel_shuffle(tokens.transpose(1, 2).reshape(1, 48, 16, 16), 4)
    assert torch.equal(restored, rgb)


def test_task_film_starts_identical_and_can_change_shared_computation():
    torch.manual_seed(19)
    features = (torch.randn(1, 256, 384), torch.randn(1, 256, 384), torch.randn(1, 64, 384))
    plain, conditioned = DenseDecoder('early', 9107), DenseDecoder('conditioned', 9107)
    for name, value in plain.state_dict().items():
        assert torch.equal(value, conditioned.state_dict()[name])
    for task in ('rgb', 'mask'):
        assert torch.equal(plain(features, task), conditioned(features, task))
    with torch.no_grad():
        conditioned.films[0].weight.fill_(.2)
    a = conditioned.represent(features, 'rgb')
    b = conditioned.represent(features, 'mask')
    assert not torch.allclose(a, b)
    # All width-independent parameters stay paired for the cheaper raw slot.
    raw = DenseDecoder('raw', 9107)
    for name, value in plain.state_dict().items():
        if name != 'projections.0.weight':
            assert torch.equal(value, raw.state_dict()[name])
