import torch
import pytest

from pathwm.evaluation.spatial_vae import color_grid_metrics


def test_color_and_grid_controls():
    colors = (
        torch.tensor([[0.8, 0.2, 0.1], [0.2, 0.7, 0.4], [0.1, 0.2, 0.8]])[
            :, :, None, None
        ]
        .expand(-1, -1, 32, 40)
        .clone()
    )
    perfect = color_grid_metrics(colors, colors)
    assert perfect["raw_mse"] == 0 and perfect["global_chroma_mse"] == 0
    gray = colors.mean(1, keepdim=True).expand_as(colors)
    bad = color_grid_metrics(gray, colors)
    assert bad["global_chroma_mse"] > 0.01 and abs(bad["chroma_gain"]) < 1e-6
    assert bad["period4"]["residual_rms"] < 1e-7
    patterned = colors.clone()
    patterned[..., ::2, ::2] += 0.2
    patterned[..., 1::2, 1::2] -= 0.2
    score = color_grid_metrics(patterned, colors)
    assert score["period2"]["residual_rms"] == pytest.approx(0.2 / (2**0.5), abs=1e-6)
    assert score["period4"]["residual_rms"] == pytest.approx(0.2 / (2**0.5), abs=1e-6)
    biased = color_grid_metrics(patterned + 0.1, colors)
    assert biased["period4"]["residual_rms"] == pytest.approx(
        score["period4"]["residual_rms"], abs=1e-6
    )


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_continuation_freezes_the_other_component(tmp_path, part):
    import numpy as np
    from experiments.spatial_vae import train, hierarchy_settings
    from pathwm.models.spatial_vae_v2 import HierarchicalVAE
    from pathwm.models.spatial_vae import SpatialVAE
    from pathwm.data.images import Frames

    model = HierarchicalVAE(stem_channels=4, channels=[8, 12], latent_channels=2)
    source = tmp_path / "source.pt"
    model.save(source)
    data = Frames(
        np.random.default_rng(7).integers(0, 256, (8, 12, 16, 3), dtype=np.uint8),
        range(8),
    )
    config = dict(hierarchy_settings("C", 0.1), steps=4, batch_size=2, disk_free_gib=0)
    train(tmp_path / "fit", data, data, config, source=source, trainable=part)
    fitted = SpatialVAE.load(tmp_path / "fit/weights.pt")
    before = model.state_dict()
    after = fitted.state_dict()
    for key in before:
        if not key.startswith(part + "."):
            assert torch.equal(before[key], after[key]), key
    assert any(
        not torch.equal(before[k], after[k]) for k in before if k.startswith(part + ".")
    )
