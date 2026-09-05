"""LeWM components with explicit tensor interfaces."""
from torch import nn

class WorldModel(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector, pred_proj):
        super().__init__()
        self.encoder, self.predictor, self.action_encoder = encoder, predictor, action_encoder
        self.projector, self.pred_proj = projector, pred_proj

    def encode(self, pixels):
        """[B,T,C,H,W] normalized pixels -> [B,T,D] states."""
        raise NotImplementedError

    def predict(self, states, actions):
        """[B,T,D] states and [B,T,K*A] actions -> next states [B,T,D]."""
        raise NotImplementedError

    def rollout(self, context, actions):
        """H context states and H+N-1 actions -> N future states."""
        raise NotImplementedError


def build_model(config=None):
    raise NotImplementedError
