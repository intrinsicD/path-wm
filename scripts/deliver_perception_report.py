"""Assemble and browser-verify the overnight report with the established adapter."""
import fcntl
import json
from world_model.curriculum.perception_cache import ROOT
from world_model.curriculum.perception_report import build
from world_model.pusht.checkpoints import json_atomic
from viewer.dashboard import _deliver_portable_artifact,find_portable_artifact_builder


def main():
    artifact=build().resolve(); html=artifact.with_name(artifact.name.replace('.artifact.json','.html'))
    with (ROOT/'dashboard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=_deliver_portable_artifact(artifact,html,find_portable_artifact_builder())
        json_atomic(html.with_suffix('.receipt.json'),receipt)
        if receipt.get('stages',{}).get('verification')!='passed':
            raise RuntimeError('Report browser verification remains incomplete: '+json.dumps(receipt))
    print(json.dumps(dict(html=str(html),verification=receipt['stages']['verification'],counts=receipt['counts'])))


if __name__=='__main__': main()
