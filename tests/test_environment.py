import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import numpy as np
from third_party.swm.pusht import PushT


def test_pinned_pusht_reset_goal_and_step_are_compatible():
    env=PushT(resolution=96,relative=False)
    state=np.array([100.,100.,256.,256.,.3,0.,0.])
    obs,info=env.reset(seed=42,options={'state':state,'goal_state':state})
    assert info['goal'].shape==(96,96,3)
    assert obs['state'].shape==(7,)
    success,distance=env.eval_state(obs['state'],obs['state'])
    assert success and distance==0
    nxt,reward,_,_,_=env.step(np.array([100.,100.]))
    assert np.isfinite(nxt['state']).all() and np.isfinite(reward)
    env.close()
