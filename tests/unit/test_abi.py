"""world_state/abi.py: the typed view of docs/abi/abi_v1.yaml and its checks.

Essential because a silent shape or dtype mismatch would invalidate every
stitching result (CLAUDE.md §4).
"""
import pytest
import torch
import yaml

from world_state.abi import DEFAULT_SPEC, ABI, ABIError, load_abi


@pytest.fixture(scope="module")
def abi() -> ABI:
    return load_abi()


def test_load_matches_yaml(abi):
    spec = yaml.safe_load(DEFAULT_SPEC.read_text())
    assert abi.version == spec["abi_version"]
    assert abi.n_tokens == spec["state"]["grid_tokens"] + spec["state"]["global_tokens"]
    assert abi.dim == spec["state"]["dim"]
    assert abi.dtype is torch.bfloat16
    assert abi.stats_dtype is torch.float32
    assert abi.register_options == tuple(spec["registers"]["count_options"])
    assert abi.max_chunk == spec["delta_t"]["max_chunk"]


def test_check_state_accepts_abi_layout(abi):
    abi.check_state(torch.zeros(2, abi.n_tokens, abi.dim, dtype=abi.dtype))


@pytest.mark.parametrize(
    "bad",
    [
        lambda a: torch.zeros(2, a.n_tokens - 1, a.dim, dtype=a.dtype),   # missing a token
        lambda a: torch.zeros(2, a.n_tokens, a.dim + 1, dtype=a.dtype),   # wrong width
        lambda a: torch.zeros(a.n_tokens, a.dim, dtype=a.dtype),          # no batch dim
        lambda a: torch.zeros(2, a.n_tokens, a.dim, dtype=torch.float32), # wrong dtype
    ],
)
def test_check_state_rejects(abi, bad):
    with pytest.raises(ABIError):
        abi.check_state(bad(abi))


def test_check_actions(abi):
    abi.check_actions(torch.zeros(3, abi.action_dims))                    # Environment.step / inverse head form
    abi.check_actions(torch.zeros(3, 1, abi.action_dims))                 # shortest chunk
    abi.check_actions(torch.ones(3, abi.max_chunk, abi.action_dims))     # longest chunk, at the range bound
    with pytest.raises(ABIError):
        abi.check_actions(torch.zeros(3, abi.max_chunk + 1, abi.action_dims))
    with pytest.raises(ABIError):
        abi.check_actions(torch.full((3, 2, abi.action_dims), 1.5))
    with pytest.raises(ABIError):
        abi.check_actions(torch.zeros(3))
    with pytest.raises(ABIError):
        abi.check_actions(torch.zeros(3, 2, abi.action_dims + 1))


def test_check_registers(abi):
    for n in abi.register_options:
        abi.check_registers(n)
    with pytest.raises(ABIError):
        abi.check_registers(3)
