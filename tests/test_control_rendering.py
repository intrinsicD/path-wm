"""Qualitative panels must reproduce saved outcomes, not invent successful frames."""
import json
import h5py
import numpy as np
import pytest
from scripts.render_prepared_control import render
from world_model.eval_tworoom import _environment, run_actions


def test_tworoom_saved_actions_and_protocol_are_verified_before_rendering(tmp_path):
    state, target = np.array([50, 50], np.float32), np.array([55, 50], np.float32)
    env = _environment(state, target, 42)
    source_path = tmp_path / 'source.h5'
    with h5py.File(source_path, 'w') as source:
        source['pixels'] = np.repeat(env.render()[None], 26, axis=0)
        source['proprio'] = np.stack([state] * 25 + [target])
        source['action'] = np.zeros((26, 2), np.float32)
    env.close()
    manifest = dict(dataset=dict(name='tworoom', path=str(source_path)), goal_offset=25,
                    budget=50, cases=[dict(episode=0, start=0, row=0)], solver_and_reset_seeds=[42],
                    horizon=5, action_block=5, receding_horizon=5, samples=300, iterations=30, elites=30)
    record = run_actions(state, target, np.zeros((1, 2)), budget=50, seed=42)
    dirs = [tmp_path / 'local', tmp_path / 'released']
    for i, directory in enumerate(dirs):
        directory.mkdir()
        (directory / 'manifest.json').write_text(json.dumps(dict(manifest, released=bool(i), step=None if i else 123)))
        (directory / 'cases.jsonl').write_text(json.dumps(record) + '\n')
        np.save(directory / 'actions_0.npy', np.zeros((1, 2), np.float32))
    output = tmp_path / 'panel'
    render(*dirs, output)
    result = json.loads((output / 'manifest.json').read_text())
    assert result['case_index'] == 0 and result['goal_offset'] == 25
    assert all(m['success'] and m['initial_success'] and m['steps'] == 1 for m in result['measurements'].values())
    assert any('123' in label for label in result['measurements'])
    assert (output / 'case_0_rollouts.png').is_file()
    changed = dict(record, final_state=[123., 50.])
    (dirs[1] / 'cases.jsonl').write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='recorded control outcome'):
        render(*dirs, tmp_path / 'bad_state')
    (dirs[1] / 'manifest.json').write_text(json.dumps(dict(manifest, goal_offset=100)))
    with pytest.raises(ValueError, match='protocol'):
        render(*dirs, tmp_path / 'bad_protocol')
