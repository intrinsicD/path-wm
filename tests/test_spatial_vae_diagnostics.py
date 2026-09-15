import torch
import pytest

from pathwm.evaluation.spatial_vae import color_grid_metrics


def test_color_and_grid_controls():
    colors = torch.tensor([[.8, .2, .1], [.2, .7, .4], [.1, .2, .8]])[:, :, None, None].expand(-1, -1, 32, 40).clone()
    perfect = color_grid_metrics(colors, colors)
    assert perfect['raw_mse'] == 0 and perfect['global_chroma_mse'] == 0
    gray = colors.mean(1, keepdim=True).expand_as(colors)
    bad = color_grid_metrics(gray, colors)
    assert bad['global_chroma_mse'] > .01 and abs(bad['chroma_gain']) < 1e-6
    assert bad['period4']['residual_rms'] < 1e-7
    patterned = colors.clone()
    patterned[..., ::2, ::2] += .2
    patterned[..., 1::2, 1::2] -= .2
    score = color_grid_metrics(patterned, colors)
    assert score['period2']['residual_rms'] == pytest.approx(.2/(2**.5), abs=1e-6)
    assert score['period4']['residual_rms'] == pytest.approx(.2/(2**.5), abs=1e-6)
    biased = color_grid_metrics(patterned + .1, colors)
    assert biased['period4']['residual_rms'] == pytest.approx(score['period4']['residual_rms'], abs=1e-6)
