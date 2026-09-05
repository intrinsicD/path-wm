"""Published LeWM components behind explicit, replaceable tensor interfaces.

Native module names match the released checkpoint. No pretrained encoder, EMA,
inverse head, decoder or additional objective is introduced.
"""
import torch
from torch import nn
from transformers import ViTConfig, ViTModel
from third_party.lewm.module import ARPredictor, Embedder, MLP


class WorldModel(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector, pred_proj):
        super().__init__()
        self.encoder, self.predictor, self.action_encoder = encoder, predictor, action_encoder
        self.projector, self.pred_proj = projector, pred_proj

    def features(self, pixels):
        return self.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]

    def encode(self, pixels):
        """[B,T,C,H,W] normalized pixels -> [B,T,D] states."""
        b, t = pixels.shape[:2]
        return self.projector(self.features(pixels.flatten(0, 1).float())).reshape(b, t, -1)

    def predict(self, states, actions):
        """Each action block begins at its paired observation's time."""
        preds = self.predictor(states, self.action_encoder(actions))
        return self.pred_proj(preds.flatten(0, 1)).reshape_as(preds)

    def rollout(self, context, actions):
        """H observed states and H+N-1 action blocks -> N future states.

        The first H-1 action blocks are the known historical transitions.
        No future observation is read while rolling out.
        """
        h = context.shape[1]
        if actions.shape[1] < h:
            raise ValueError('At least H action blocks are required')
        states = context
        future = []
        history = self.predictor.pos_embedding.shape[1]
        for end in range(h, actions.shape[1] + 1):
            start = max(0, end - history)
            nxt = self.predict(states[:, start:end], actions[:, start:end])[:, -1:]
            future.append(nxt)
            states = torch.cat((states, nxt), dim=1)
        return torch.cat(future, dim=1)


def build_model(config=None):
    """Constructor injection supports swaps; this builder selects the baseline."""
    c = dict(width=192, image_size=224, patch_size=14, encoder_depth=12,
             encoder_heads=3, predictor_depth=6, predictor_heads=16,
             head_dim=64, mlp_dim=2048, projector_dim=2048, history=3,
             action_dim=2, frameskip=5, dropout=0.1)
    if config:
        unknown = set(config) - set(c)
        if unknown:
            raise ValueError(f'Unknown model options: {unknown}')
        c.update(config)
    d = c['width']
    # Matches stable-pretraining vit_hf(tiny, pretrained=False), including defaults.
    encoder = ViTModel(ViTConfig(hidden_size=d, num_hidden_layers=c['encoder_depth'],
        num_attention_heads=c['encoder_heads'], intermediate_size=d*4,
        image_size=c['image_size'], patch_size=c['patch_size']),
        add_pooling_layer=False, use_mask_token=False)
    encoder.config.interpolate_pos_encoding = True
    return WorldModel(encoder,
        ARPredictor(num_frames=c['history'], input_dim=d, hidden_dim=d, output_dim=d,
            depth=c['predictor_depth'], heads=c['predictor_heads'], dim_head=c['head_dim'],
            mlp_dim=c['mlp_dim'], dropout=c['dropout'], emb_dropout=0.0),
        Embedder(input_dim=c['action_dim']*c['frameskip'], emb_dim=d),
        MLP(d, c['projector_dim'], d, norm_fn=nn.BatchNorm1d),
        MLP(d, c['projector_dim'], d, norm_fn=nn.BatchNorm1d))
