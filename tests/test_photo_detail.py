"""Causal stage access, rank/nullspace and exact regression controls."""

import torch


def test_ridge_matches_primal_and_reload_uses_training_statistics(tmp_path):
    from pathwm.models.photo_probe import RidgeReader

    torch.manual_seed(61)
    x = torch.randn(40, 7, dtype=torch.float64)
    y = x @ torch.randn(7, 5, dtype=torch.float64) + 2
    q = torch.randn(9, 7, dtype=torch.float64)
    m = RidgeReader.fit(x, y, ridge=0.01, kernel="linear")
    z = (x - m["mean"]) / m["std"]
    v = (q - m["mean"]) / m["std"]
    w = torch.linalg.solve(
        z.T @ z + 0.01 * 7 * torch.eye(7, dtype=z.dtype), z.T @ (y - m["target_mean"])
    )
    expected = v @ w + m["target_mean"]
    torch.testing.assert_close(
        RidgeReader.predict(m, q), expected, atol=1e-10, rtol=1e-10
    )
    torch.save(m, tmp_path / "probe.pt")
    r = torch.load(tmp_path / "probe.pt", weights_only=True)
    torch.testing.assert_close(
        RidgeReader.predict(r, q), expected, atol=1e-10, rtol=1e-10
    )
    a = RidgeReader.predict(r, q[:1])
    b = RidgeReader.predict(r, torch.cat([q[:1], q[1:] + 1000]))[:1]
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


