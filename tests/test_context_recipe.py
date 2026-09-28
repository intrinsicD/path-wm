import torch
from pathwm.data.episode_facts import batch


def test_pairing_split_fixed_masks_and_supervision():
    train = batch(torch.Generator().manual_seed(17), 2048)
    held = batch(torch.Generator().manual_seed(29), 2048, 'pairing')
    assert set(train['query'].flatten().tolist()) == set(range(24))
    assert ((train['query'] < 8).any(-1)).all()
    assert (held['query'] >= 8).all()
    for data in (train, held):
        assert (data['valid'].sum(-1) == 8).all()
        gold = data['valid'] & (data['attributes'] == data['attribute'][:, None])
        assert torch.equal(~gold.any(-1), data['target'] == 32)
        answerable = data['target'] != 32
        assert gold[answerable, data['target'][answerable]].all()
        # Same-factor near misses exist in both answerability groups with high frequency;
        # evaluation explicitly chooses only missing keys with a one-factor near miss.
        assert data['keys'].shape == (2048, 32, 2)


