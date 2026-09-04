"""contracts.py imports on its own and its Protocols are runtime-checkable.

Step-1 rule (CLAUDE.md §2): the interface file has no implementation and is
importable by every test, so a red conformance test fails on the missing
implementation, never on an import error.
"""
import torch

import contracts


def test_every_contract_is_exported():
    for name in [
        "Environment", "Encoder", "Adapter", "Updater", "Predictor",
        "TemporalObservation", "EvidenceTokens", "ActionSequence", "ActionTokens",
        "EvidenceEncoder", "EvidenceAdapter", "ActionAdapter", "WorldPredictorV2",
        "BeliefUpdaterV2", "WorldModelCore",
        "InverseDynamics", "Critic", "Constraints", "Planner", "DebugDecoder", "Goal",
    ]:
        assert hasattr(contracts, name), name


def test_runtime_check_needs_the_method_and_the_attribute():
    class P:
        n_registers = 0

        def predict(self, W, actions, delta_t):
            return W

    class NoRegisters:
        def predict(self, W, actions, delta_t):
            return W

    assert isinstance(P(), contracts.Predictor)
    assert not isinstance(NoRegisters(), contracts.Predictor)
    assert not isinstance(P(), contracts.Encoder)


def test_goal_is_batched_and_defaults_to_all_tokens():
    g = contracts.Goal(W_G=torch.zeros(1, 65, 192, dtype=torch.bfloat16))
    assert g.W_G.ndim == 3 and g.mask is None
