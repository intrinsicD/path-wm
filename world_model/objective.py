"""LeWM's prediction MSE plus per-time SIGReg; both encoder branches learn."""
from third_party.lewm.module import SIGReg


def loss_from_embeddings(model, embeddings, actions, regularizer, weight=0.09):
    prediction = model.predict(embeddings[:, :-1], actions[:, :-1])
    pred_loss = (prediction - embeddings[:, 1:]).square().mean()
    sigreg_loss = regularizer(embeddings.transpose(0, 1))
    return dict(loss=pred_loss + weight * sigreg_loss,
                pred_loss=pred_loss, sigreg_loss=sigreg_loss)
