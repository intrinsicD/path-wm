"""PushT evaluation must preserve causal inputs and physical denominators."""
from types import SimpleNamespace
import inspect
import math

import numpy as np
import pytest
import torch

from world_model.paddle.types import ObservationLatent, PlanningState
from world_model.pusht.evaluation import (
    matched_rollout, physical_errors, summarize_prediction_records,
    run_control_case, select_case_indices,
)


def latent(value=1.):
    return ObservationLatent(torch.full((1,256,64),value), torch.full((1,64,64),value))


class Updater:
    def __call__(self, memory, observation, action):
        return memory + observation.fine[:,0,:1] + action.sum(-1,keepdim=True)


class Predictor:
    def __call__(self, observation, memory, action):
        value=observation.fine[:,0,:1]+memory[:,:1]+action[:,:1]
        return ObservationLatent(value[:,:,None].expand(-1,256,64).clone(),
                                 value[:,:,None].expand(-1,64,64).clone())


def test_matched_rollout_never_receives_future_targets_and_updates_predicted_memory():
    state=PlanningState(latent(),torch.full((1,128),2.))
    before=state.clone()
    result=matched_rollout({'P':Predictor(),'U':Updater()},state,torch.tensor([[[.1,.2],[.3,.4]]]))
    assert list(inspect.signature(matched_rollout).parameters)==['system','initial','actions']
    assert result['prediction'][1].observation.fine[0,0,0].item()==pytest.approx(8.8)
    assert result['copy'][1].observation.fine[0,0,0].item()==1.
    assert result['reset'][0].observation.fine[0,0,0].item()==pytest.approx(.1)
    assert torch.equal(state.memory,before.memory)
    assert torch.equal(state.observation.fine,before.observation.fine)


def test_wrapped_pose_error_and_motion_mask_have_distinct_denominators():
    angle=.01
    pose=np.array([.2,.3,.4,.5,math.sin(-angle),math.cos(-angle)])
    truth=np.array([.2*512,.3*512,.4*512,.5*512,angle])
    estimate=np.r_[pose,np.ones(5)]
    first=physical_errors(pose,estimate,truth,None)
    second=physical_errors(pose,estimate,truth,np.zeros(5))
    assert first['h_angle_abs_error_deg']==pytest.approx(math.degrees(.02))
    assert first['r_motion_abs_error'] is None
    records=[dict(method='actual',horizon=0,latent_error=0.,image_mse=0.,**row) for row in (first,second)]
    summary=summarize_prediction_records(records)['actual']['0']
    assert summary['count']==2 and summary['motion_count']==1
    assert summary['r_motion_mae']==pytest.approx([512.,512.,512.,512.,math.pi])


def test_case_selection_uses_earliest_unsolved_index_and_only_requested_split():
    candidates=[{'episode':0,'group_id':0,'index':8,'split':'test','block_distance':42.,'angle_error':0.},
                {'episode':0,'group_id':0,'index':3,'split':'test','block_distance':41.,'angle_error':0.},
                {'episode':0,'group_id':0,'index':2,'split':'test','block_distance':21.,'angle_error':0.},
                {'episode':1,'group_id':1,'index':2,'split':'train','block_distance':90.,'angle_error':0.},
                {'episode':2,'group_id':2,'index':2,'split':'test','block_distance':0.,'angle_error':0.}]
    selected,audit=select_case_indices(candidates,split='test',count=20,seed=3107)
    assert [(r['episode'],r['index']) for r in selected]==[(0,3)]
    assert audit['candidate_windows']==4 and audit['eligible_windows']==2
    assert audit['eligible_groups']==1 and audit['selected_cases']==1


def test_control_executes_one_primitive_action_and_never_passes_oracle_to_planner():
    instances=[];calls=[]
    class Env:
        def __init__(self):self.actions=[];self.pose=np.zeros(5);instances.append(self)
        def reset(self,pose5_world=None,goal_pose5_world=None,seed=0):return np.zeros((64,64,3),np.uint8),{}
        def set_goal(self,pose):self.goal=pose
        def step(self,action):
            self.actions.append(np.asarray(action).copy())
            self.pose[2:4]=100. if len(self.actions)==3 else 0.
            return np.zeros((64,64,3),np.uint8),0.,False,False,{}
        def close(self):pass
        @property
        def hold_action(self):return np.zeros(2)
    class Planner:
        def __init__(self,*args,**kwargs):pass
        def prepare_goal(self,observation):
            assert isinstance(observation,ObservationLatent)
            return 'encoded image goal'
        def plan(self,state,goal):
            assert isinstance(state,PlanningState) and goal=='encoded image goal'
            calls.append(state.clone())
            return SimpleNamespace(action=torch.tensor([.1,.2]),sequence=torch.full((5,2),.9),cost=0.,stats={},states=[])
    pose=np.array([0.,0.,0.,0.,0.,1.])
    system={'E':lambda x:latent(0.),'U':Updater(),'P':Predictor(),
            'H':lambda s:torch.tensor(pose)[None], 'R':lambda m:torch.tensor(np.r_[pose,np.zeros(5)])[None]}
    public={'case_id':'case-test','seed':1,'reset_pose':[0.]*5,'prefix_actions':[[.3,.4],[.4,.5]],
            'goal_frame':np.zeros((64,64,3),np.uint8),'source_episode':9,'group_id':9,'start_index':2}
    oracle={'goal_pose':[0.,0.,100.,100.,0.],'future_actions':[[.99,.99]]*5}
    result=run_control_case(system,public,oracle,'learned',{'control_max_steps':2},env_factory=Env,planner_factory=Planner)
    assert len(calls)==2 and result['episode_length']==2
    assert result['success'] is False and result['any_time_success'] is True
    np.testing.assert_allclose(instances[0].actions[-2:],[[.1,.2],[.1,.2]])
    assert len(instances[0].actions)==4  # Two observed prefix actions, then two decisions.
