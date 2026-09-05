import copy
import torch
from torch import nn
from third_party.lewm.jepa import JEPA
from third_party.lewm.module import SIGReg
from world_model.model import build_model
from world_model.training import backward_batch


def tiny():
    return build_model(dict(width=12, image_size=28, encoder_depth=1, encoder_heads=3,
        predictor_depth=1, predictor_heads=2, head_dim=4, mlp_dim=24,
        projector_dim=24, dropout=0.0))


def test_native_reference_encode_predict_rollout():
    model=tiny().eval()
    ref=JEPA(*(copy.deepcopy(getattr(model,k)) for k in
               ['encoder','predictor','action_encoder','projector','pred_proj'])).eval()
    pixels=torch.randn(2,3,3,28,28)
    actions=torch.randn(2,6,10)
    with torch.no_grad():
        native=ref.encode(dict(pixels=pixels,action=actions[:,:3]))
        z=model.encode(pixels)
        torch.testing.assert_close(z,native['emb'],atol=0,rtol=0)
        torch.testing.assert_close(model.predict(z,actions[:,:3]),
                                   ref.predict(z,native['act_emb']),atol=0,rtol=0)
        expected=ref.rollout({'pixels':pixels[:,None]},actions[:,None])['predicted_emb'][:,0,3:]
        torch.testing.assert_close(model.rollout(z,actions),expected,atol=0,rtol=0)


def test_encoder_recomputation_preserves_all_gradients_and_batchnorm():
    torch.manual_seed(4)
    a=tiny().train()
    # Exercise action gradients too: baseline AdaLN gates start at exactly zero.
    for name,p in a.named_parameters():
        if 'adaLN_modulation.1' in name:
            with torch.no_grad(): p.normal_(0,.02)
    b=copy.deepcopy(a)
    pixels=torch.randn(3,4,3,28,28)
    actions=torch.randn(3,4,10)
    reg=SIGReg(num_proj=16)
    torch.manual_seed(99)
    x=backward_batch(a,pixels,actions,reg,encoder_chunk=0)
    torch.manual_seed(99)
    y=backward_batch(b,pixels,actions,reg,encoder_chunk=2)
    for k in x: torch.testing.assert_close(x[k],y[k],rtol=1e-5,atol=1e-6)
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-3,atol=1e-5,msg=name)
    for p,q in zip(a.buffers(),b.buffers()):
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
