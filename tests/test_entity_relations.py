import copy
import pytest
import torch
from pathwm.models.entity_relations import RelationKey, EntityRelationMemory
from pathwm.models.entity_state import EntityInteractionCell
from tests.test_entity_memory import Scorer


def key_model():
    model=RelationKey(width=8)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(8));model.encoder.bias.zero_()
        model.decoder.weight.copy_(torch.eye(8));model.decoder.bias.zero_()
    return model


def test_relation_replacement_and_old_retry():
    matcher,cell,key=Scorer(),EntityInteractionCell(),key_model()
    store=EntityRelationMemory(matcher,cell,key)
    a,b,c=torch.eye(8)[:3]
    for t,q in enumerate((a,b,c)):store.observe(str(t),q,t,[1,0,0,0])
    latents=copy.deepcopy(store.state.latents)
    bound=store.bind('bind',a,b,3)
    assert store.state.latents==latents
    store.observe('toggle',b,4,[0,0,1,0])
    before=store.snapshot()
    read=store.recall('read',a,5)
    assert read['source_id']==1
    assert store.state.latents[1:]==before['state']['latents'][1:]
    store.bind('replace',a,c,6)
    snapshot=store.snapshot()
    assert store.recall('read',a,5)==read
    assert store.bind('bind',a,b,3)==bound
    assert store.snapshot()==snapshot
    restored=EntityRelationMemory.restore(matcher,cell,key,snapshot)
    assert restored.recall('next',a,7)['source_id']==2
    with pytest.raises(ValueError,match='Conflicting'):
        store.bind('bind',a,c,3)
    assert store.snapshot()==snapshot


def test_relation_failure_and_snapshot_compatibility():
    matcher,cell,key=Scorer(),EntityInteractionCell(),key_model()
    store=EntityRelationMemory(matcher,cell,key)
    a,b,c=torch.eye(8)[:3]
    store.observe('a',a,0,[1,0,0,0]);store.observe('b',b,1,[0,1,0,0])
    before=store.snapshot()
    with pytest.raises(LookupError):store.recall('missing',a,2)
    with pytest.raises(LookupError):store.bind('unknown',a,c,2)
    assert store.snapshot()==before
    store.bind('bind',a,b,2)
    snapshot=store.snapshot()
    wrong=copy.deepcopy(key)
    with torch.no_grad():wrong.decoder.bias.add_(1)
    with pytest.raises(ValueError):EntityRelationMemory.restore(matcher,cell,wrong,snapshot)
    bad=copy.deepcopy(snapshot);bad['relations']['99']=[0.]*8
    with pytest.raises(ValueError):EntityRelationMemory.restore(matcher,cell,key,bad)
    class Broken(torch.nn.Module):
        def forward(self,x):raise RuntimeError('injected')
    store.state.cell.interaction=Broken()
    before=store.snapshot()
    with pytest.raises(RuntimeError,match='injected'):store.recall('failure',a,3)
    assert store.snapshot()==before
