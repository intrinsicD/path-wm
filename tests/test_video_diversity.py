import json

import pytest
import torch


def test_manifest_rejects_confirmation_subject_and_file_leakage(tmp_path):
    from experiments.video_order import load_sources

    groups = {}
    for i, split in enumerate(('train', 'validation', 'evaluation', 'confirmation')):
        path = tmp_path / (split + '.mp4')
        path.write_bytes(bytes([i]) * 8)
        groups[split] = [dict(id=split, subject=str(i), path=str(path), frames=4)]
    manifest = tmp_path / 'sources.json'
    def load():
        manifest.write_text(json.dumps(dict(schema=1, splits=groups)))
        return load_sources(manifest, tmp_path)
    data = load()
    assert data['train'][0]['frames'] == 4
    assert data['confirmation'][0]['subject'] == '3'
    groups['confirmation'][0]['subject'] = '0'
    with pytest.raises(ValueError, match='subject'):
        load()
    groups['confirmation'][0]['subject'] = '3'
    (tmp_path / 'confirmation.mp4').write_bytes((tmp_path / 'train.mp4').read_bytes())
    with pytest.raises(ValueError, match='duplicate'):
        load()
    (tmp_path / 'confirmation.mp4').write_bytes(b'unique')
    groups['train'][0]['frames'] = True
    with pytest.raises(ValueError, match='frames'):
        load()


def test_source_macro_weights_clips_equally_not_by_pair_count():
    from experiments.video_order import source_metrics

    labels = torch.tensor([[0, 1]]).expand(4, -1)
    logits = torch.tensor([[[2., 0.], [0., 2.]]]).expand(4, -1, -1).clone()
    logits[-1] = logits[-1].flip(-1)
    data = dict(source_index=torch.tensor([0, 0, 0, 1]), source_ids=['A', 'B'])
    score = source_metrics(logits, labels, data)
    assert score['by_source']['A']['accuracy'] == 1
    assert score['by_source']['B']['accuracy'] == 0
    assert score['source_macro_accuracy'] == .5
    assert score['source_macro_pair_accuracy'] == .5
