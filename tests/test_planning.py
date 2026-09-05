import torch
from world_model.planning import cem


def test_cem_improves_known_dynamics_and_verifies_executed_mean():
    observed=[]
    def cost(actions):
        observed.append(actions.clone())
        return (actions.sum(1)-2).square().sum(1)
    actions,stats=cem(cost,3,2,torch.device('cpu'),samples=100,iterations=8,elites=10)
    assert stats['terminal_cost']<.01
    assert stats['candidate_evaluations']==801
    torch.testing.assert_close(observed[-1][0],actions)
