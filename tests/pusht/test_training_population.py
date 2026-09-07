"""A changed perception population must be explicit and resumable."""
import json
import sys
from types import ModuleType

import pytest

from world_model.pusht import training, checkpoints
from tests.pusht.test_training_audit import _tiny_perception


def test_mixed_population_counts_and_identity_are_saved_before_completed_resume(tmp_path, monkeypatch):
    config = _tiny_perception(monkeypatch)
    config['perception_supplement'] = {'directory':'fixture','fingerprint':'supplement-a'}
    config['training']['perception'].update(updates=4,batch_size=4)
    identity = {'fingerprint':'supplement-a','count':4}
    module = ModuleType('world_model.pusht.perception_data')
    class Supplement:
        def __init__(self,path,expected_fingerprint):
            if expected_fingerprint != identity['fingerprint']:
                raise ValueError('supplement fingerprint changed')
            self.manifest = dict(identity)
    class Mixed:
        def __init__(self,source,supplement):
            self.dataset,self.lengths=source.dataset,source.lengths
            self.frames=list(range(8))
    module.SupplementFrames,module.MixedPerceptionSamples=Supplement,Mixed
    monkeypatch.setitem(sys.modules,'world_model.pusht.perception_data',module)
    run=tmp_path/'run'
    result=training.train_perception(config,'unused',run)
    saved=checkpoints.read_checkpoint(run/'last.pt')
    manifest=json.loads((run/'pusht_manifest.json').read_text())
    population=saved['training_population']
    assert population['source_frames']==population['supplement_frames']==4
    assert population['supplement_fingerprint']=='supplement-a'
    assert manifest['training_population']==result['training_population']==population
    counts=saved['population_presentations']
    assert counts['source']+counts['supplement']==16
    assert counts['source']>0 and counts['supplement']>0
    assert result['population_presentations']==counts
    before=(run/'training.jsonl').read_bytes()
    assert training.train_perception(config,'unused',run,resume=True)['population_presentations']==counts
    assert (run/'training.jsonl').read_bytes()==before
    identity['fingerprint']='tampered-supplement'
    with pytest.raises(ValueError,match='fingerprint|population'):
        training.train_perception(config,'unused',run,resume=True)
