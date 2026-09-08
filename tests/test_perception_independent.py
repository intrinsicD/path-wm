import torch
from world_model.curriculum.perception_decoder import DenseDecoder
from world_model.curriculum.perception_independent import IndependentDecoder


def test_independent_branches_match_shared_initial_functions_without_shared_storage():
    torch.manual_seed(73); source=DenseDecoder('early',9107); model=IndependentDecoder(9107)
    features=(torch.randn(2,256,384),torch.randn(2,256,384),torch.randn(2,64,384))
    for a,b in zip(source.both(features),model.both(features)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    a={p.data_ptr() for p in model.heads['rgb'].parameters()}
    b={p.data_ptr() for p in model.heads['mask'].parameters()}
    assert a.isdisjoint(b)


def test_optimizing_one_output_cannot_update_the_other_branch():
    torch.manual_seed(79); model=IndependentDecoder(9107)
    features=(torch.randn(2,256,384),torch.randn(2,256,384),torch.randn(2,64,384))
    before={n:p.detach().clone() for n,p in model.heads['mask'].state_dict().items()}
    with torch.no_grad(): old=model(features,'mask')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    model(features,'rgb').square().mean().backward()
    assert all(p.grad is None for p in model.heads['mask'].parameters())
    optimizer.step()
    for n,p in model.heads['mask'].state_dict().items():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    with torch.no_grad():torch.testing.assert_close(model(features,'mask'),old,rtol=0,atol=0)
