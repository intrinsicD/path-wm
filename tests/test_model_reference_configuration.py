"""Rerun existing assertions on actual configured modules, without surrogate models.

Reference: experiments.multimodal.build_model defaults (width32; categorical
context16/evidence8/8x8 codes, memory32/8/16). These are software checks with
untrained weights, not a fully trained deployment model or new quality evidence.
Native-resolution R2/checkpoint and image256 executions have separate receipts in
runs/reviews/actual_model_reruns_20260924. No reduced configuration is run here.
"""
import inspect
from dataclasses import replace
import copy

import pytest
import torch

from experiments.multimodal import build_model
from pathwm.models.modalities import Observation
from pathwm.models.multiscale import MultiScaleImageEncoder, MultiScaleAudioEncoder, MultiScaleTextEncoder
from pathwm.evaluation.agent import plan
from pathwm.training.improvement import try_improvement
from pathwm.io import state_hash
from tests import test_belief as belief_checks
from tests import test_multimodal as gaussian_checks
from tests import test_token_budget as event_checks
from tests.test_runs import equal_tree


def reference(kind='belief'):
    torch.manual_seed(71)
    model = build_model(state_model=kind).eval()
    assert model.width == 32
    assert isinstance(model.encoders['image'], MultiScaleImageEncoder)
    assert isinstance(model.encoders['video'], MultiScaleImageEncoder)
    assert isinstance(model.encoders['audio'], MultiScaleAudioEncoder)
    assert isinstance(model.encoders['text'], MultiScaleTextEncoder)
    if kind == 'belief':
        assert (model.memory.capacity, model.memory.block_size, model.memory.block_capacity) == (32,8,16)
    return model


BELIEF_ASSERTIONS = [
    'test_event_union_deduplicates_without_reapplying_action',
    'test_live_and_imagined_share_transition_and_no_observation_creates_no_evidence',
    'test_source_features_are_independent_of_belief_and_hidden_values',
    'test_safe_snapshot_roundtrip_rejects_other_schemas',
    'test_planner_reuses_complete_starting_draws_across_candidates',
    'test_repeated_seal_and_planning_restore_rng_even_on_exception',
    'test_mixed_batch_mask_blocks_hidden_gradients_and_keeps_prior_code',
    'test_compression_gradients_are_bounded_and_source_compressor_has_no_belief_input',
    'test_consolidation_preserves_old_member_when_new_block_has_no_evidence',
    'test_full_belief_distribution_reaches_readers_and_compression',
    'test_memory_rejects_later_ordinals_even_at_equal_time_with_old_evidence',
]


@pytest.mark.parametrize('assertion', BELIEF_ASSERTIONS)
def test_existing_belief_assertions_at_reference_configuration(assertion, monkeypatch):
    # Replace only the small construction helper with the real recipe's reference
    # construction. The original assertions and real forward methods are unchanged.
    monkeypatch.setattr(belief_checks, 'model', reference)
    function = getattr(belief_checks, assertion)
    arguments = {'monkeypatch': monkeypatch} if 'monkeypatch' in inspect.signature(function).parameters else {}
    function(**arguments)


@pytest.mark.parametrize('assertion', [
    'test_all_modalities_outputs_and_gradients',
    'test_masks_hide_values_and_future_observations_are_rejected',
    'test_thinking_memory_imagination_and_state_roundtrip',
    'test_large_timestamps_preserve_short_intervals_and_same_time_updates',
])
def test_existing_gaussian_assertions_at_reference_configuration(assertion, monkeypatch):
    monkeypatch.setattr(gaussian_checks, 'small', lambda: reference('gaussian'))
    getattr(gaussian_checks, assertion)()


@pytest.mark.parametrize('training', [False, True])
def test_complete_event_posterior_gradients_and_rng_at_reference_size(training, monkeypatch):
    def actual_builder(**kwargs):
        kwargs['width'] = 32
        kwargs.update(memory_recent=32, memory_block=8, memory_blocks=16)
        return build_model(**kwargs)
    monkeypatch.setattr(event_checks, 'build_model', actual_builder)
    event_checks.test_complete_event_encodes_once_with_same_posterior_and_gradients(training)


@torch.no_grad()
def test_full_memory_capacity_consolidation_timestamps_and_protection():
    model = reference()
    state = model.initial_state(1)
    # Fill the REAL capacity rather than lowering capacity to trigger eviction.
    count = model.memory.capacity + model.memory.block_size*(model.memory.block_capacity+2)
    for ordinal in range(1,count+1):
        pending = model.begin_event(state,event_id=f'e{ordinal}',ordinal=ordinal,time=float(ordinal))
        state = model.commit_event(model.add_packet(pending,belief_checks.packet(f'camera{ordinal}',time=0.)))
        bank = state.memory
        assert len(bank.recent)<=32 and len(bank.staging)<8 and len(bank.compressed)<=16
    assert bank.consolidated is not None
    assert state.time.item()==count and state.observed_time.item()==0
    assert bank.consolidated.end_time.item()==0
    assert torch.equal(bank.recent[-1].h,state.h)
    assert torch.equal(bank.recent[-1].logits,state.logits)
    assert not bank.recent[-1].h.requires_grad
    assert model.memory.storage_bytes(bank)<=model.memory.capacity_bytes(32,8,8,8)
    marked=model.mark(state,author='user',detail='retain actual source')
    thought=model.think(marked,steps=4)
    assert torch.equal(thought.h,marked.h) and torch.equal(thought.logits,marked.logits)
    assert torch.equal(thought.time,marked.time) and thought.memory is marked.memory


@torch.no_grad()
def test_real_dynamics_planner_matches_direct_execution_instead_of_toy_transition():
    model=reference('gaussian')
    state=model.observe(model.initial_state(2),{'image':gaussian_checks.image_observation()},time=0)
    candidates=torch.tensor([[[[0.,0.]],[[1.,0.]],[[-1.,0.]]]]).expand(2,-1,-1,-1)
    target=model.imagine(state,candidates[:,1,0]).tokens
    before=state.to_dict(); version=state_hash(model); rng=torch.get_rng_state().clone()
    metric=lambda s:(s.tokens-target).square().mean((1,2))
    expected=torch.stack([metric(model.imagine(state,candidates[:,i,0])) for i in range(3)],1)
    result=plan(model,state,candidates,metric,lower=-1,upper=1)
    torch.testing.assert_close(result.scores,expected)
    assert torch.equal(result.indices,expected.argmin(1))
    assert result.indices.tolist()==[1,1]
    equal_tree(before,state.to_dict())
    assert state_hash(model)==version and torch.equal(torch.get_rng_state(),rng)
    with pytest.raises(ValueError,match='bounds'):
        plan(model,state,candidates*2,metric,lower=-1,upper=1)


def test_guarded_update_and_rollback_on_actual_model_output():
    # Two disposable correctness proposals, no trained checkpoint or quality claim.
    model=reference('gaussian')
    observation=gaussian_checks.image_observation(batch=1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    def loss():
        state=model.observe(model.initial_state(1),{'image':observation},time=0)
        return model.decode(state)['image'].square().mean()
    def evaluate(): return {'error':float(loss().detach())}
    def proposal():
        optimizer.zero_grad(set_to_none=True);loss().backward();optimizer.step()
    accepted=try_improvement(model,optimizer,proposal,evaluate,primary='error',min_improvement=1e-9,tolerances={'error':0.})
    assert accepted['accepted']
    before=copy.deepcopy(model.state_dict()); opt=copy.deepcopy(optimizer.state_dict()); rng=torch.get_rng_state().clone()
    def bad():
        proposal()
        with torch.no_grad():
            for parameter in model.decoders['image'].parameters(): parameter.add_(100)
        torch.rand(7)
    rejected=try_improvement(model,optimizer,bad,evaluate,primary='error',min_improvement=1e-9,tolerances={'error':0.})
    assert not rejected['accepted']
    equal_tree(before,model.state_dict());equal_tree(opt,optimizer.state_dict())
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('assertion', [
    'test_induce_is_finite_empty_padding_and_permutation_safe',
    'test_core_loss_reaches_all_trainable_core_parameters',
    'test_perception_loss_trains_multiscale_encoder_and_slots',
])
def test_existing_r1_assertions_on_full_configured_core_and_perception(assertion, monkeypatch):
    from tests import test_latent_agent as original
    from pathwm.models.latent_core import LatentCore
    from pathwm.models.slots import SlotPerception
    monkeypatch.setattr(original, 'WIDTH', 64)
    monkeypatch.setattr(original, 'tiny_core', lambda: LatentCore())
    def actual_perception(**unused_small_settings):
        return SlotPerception()  # Existing native64/7slots/3iterations/decoder32.
    monkeypatch.setattr(original, 'SlotPerception', actual_perception)
    getattr(original, assertion)()


def test_full_core_query_isolation_with_its_actual_inferred_code_shape():
    from pathwm.models.latent_core import LatentCore
    core=LatentCore().eval()
    support=torch.randn(4,6,64)
    code=core.induce(support,torch.ones(4,6,dtype=torch.bool))
    assert code.shape==(4,4,64)
    machine,a,b=(torch.randn(4,64) for _ in range(3))
    before=core.apply(machine,a,b,code)
    changed=machine.clone();changed[1:]=torch.randn(3,64)
    after=core.apply(changed,a,b,code)
    assert all(torch.equal(x[0],y[0]) for x,y in zip(before,after))


@pytest.mark.parametrize('modality', ['image','video','audio','text'])
def test_reference_multiscale_exports_and_future_masking(modality):
    model=reference()
    encoder=model.encoders[modality]
    if modality in ('image','video'):
        values=torch.randn(1,3,3,64,64)
    elif modality=='audio':
        values=torch.randn(1,3,32)
    else:
        values=torch.tensor([[3,7,11]])
    times=torch.tensor([[0.,1.,2.]])
    obs=Observation(values,times)
    before=encoder(obs)
    altered=values.clone();altered[:,2]=0
    after=encoder(Observation(altered,times))
    assert len(before.scales)==3
    torch.testing.assert_close(before.as_tokens().values,torch.cat([s.values for s in before.scales],1))
    for left,right in zip(before.scales,after.scales):
        assert left.values.shape[-1]==32
        past=left.valid & (left.times<2.)
        torch.testing.assert_close(left.values[past],right.values[past],rtol=0,atol=0)
    assert not torch.equal(before.as_tokens().values,after.as_tokens().values)


@pytest.mark.parametrize('assertion', [
    'test_life_keeps_weights_frozen_restarts_and_ignores_poisoned_hidden_fields',
    'test_planner_inputs_contain_no_environment',
    'test_prepared_once_tokens_equal_live_agent_encoding',
    'test_successful_execution_feedback_updates_only_the_bound_concept',
    'test_failed_receipts_are_kept_but_never_teach',
    'test_claim_test_enters_the_same_feedback_path',
    'test_feedback_can_be_disabled_for_the_declared_control',
    'test_feedback_does_not_stale_reads_of_unrelated_concepts',
])
def test_r1_runtime_assertions_on_actual_size_models(assertion, monkeypatch, tmp_path):
    from tests import test_latent_agent as original
    from pathwm.models.latent_core import LatentCore
    from pathwm.models.slots import SlotPerception
    monkeypatch.setattr(original,'WIDTH',64)
    def actual_models():
        torch.manual_seed(2)
        return SlotPerception().eval(),LatentCore().eval()
    monkeypatch.setattr(original,'tiny_models',actual_models)
    monkeypatch.setattr(original,'tiny_core',lambda: LatentCore())
    monkeypatch.setattr(original,'SlotPerception',lambda **kwargs:SlotPerception())
    getattr(original,assertion)(tmp_path)
