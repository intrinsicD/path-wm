"""Causal stage access, rank/nullspace and exact regression controls."""
import numpy as np
import torch


def test_ridge_matches_primal_and_reload_uses_training_statistics(tmp_path):
    from pathwm.models.photo_probe import RidgeReader
    torch.manual_seed(61)
    x=torch.randn(40,7,dtype=torch.float64)
    y=x@torch.randn(7,5,dtype=torch.float64)+2
    q=torch.randn(9,7,dtype=torch.float64)
    m=RidgeReader.fit(x,y,ridge=0.01,kernel='linear')
    z=(x-m['mean'])/m['std'];v=(q-m['mean'])/m['std']
    w=torch.linalg.solve(z.T@z+0.01*7*torch.eye(7,dtype=z.dtype),z.T@(y-m['target_mean']))
    expected=v@w+m['target_mean']
    torch.testing.assert_close(RidgeReader.predict(m,q),expected,atol=1e-10,rtol=1e-10)
    torch.save(m,tmp_path/'probe.pt')
    r=torch.load(tmp_path/'probe.pt',weights_only=True)
    torch.testing.assert_close(RidgeReader.predict(r,q),expected,atol=1e-10,rtol=1e-10)
    a=RidgeReader.predict(r,q[:1]);b=RidgeReader.predict(r,torch.cat([q[:1],q[1:]+1000]))[:1]
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)


def test_stage_cache_matches_live_encoder_storage_and_workspace():
    from tests.test_memory_output import model_fixture
    from pathwm.evaluation.photo_detail import stage_values
    from pathwm.io import state_hash
    model=model_fixture(True).requires_grad_(False).eval()
    before=state_hash(model)
    rgb=torch.rand(4,3,64,64)
    stages,controls,audit=stage_values(model,rgb,native=False)
    assert audit['stored_snapshot_exact']
    assert stages['encoder'].shape[1]>stages['observed_all'].shape[1]
    assert stages['recall_working'].shape[1]<stages['recall_all'].shape[1]
    assert all(not x.requires_grad for x in stages.values())
    assert torch.equal(controls['grid'],torch.nn.functional.avg_pool2d(rgb,4))
    assert state_hash(model)==before


def test_patch_nullspace_and_spatial_error_are_numeric_not_visual_guesses():
    from pathwm.evaluation.photo_detail import patch_audit, grid_score
    layer=torch.nn.Conv2d(3,4,4,stride=4)
    audit=patch_audit(layer)
    w=layer.weight.detach().double().flatten(1)
    null=torch.tensor(audit['null_direction'],dtype=torch.float64)
    assert (w@null).abs().max()<1e-10
    assert audit['input_dimensions']==48 and audit['null_dimensions']>=44
    target=torch.zeros(2,3,16,16);target[:,:,:,8:]=1
    correct=grid_score(target,target)
    wrong=grid_score(target.flip(-1),target)
    assert correct['rgb_mse']==0 and wrong['rgb_mse']==1
