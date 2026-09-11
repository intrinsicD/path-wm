import copy
import pytest
import torch
from pathwm.models.entity_state import EntityInteractionCell, EntityStateMemory
from tests.test_entity_memory import Scorer


def test_lookup_is_read_only_and_rejects_unknown():
    store = EntityStateMemory(Scorer(), EntityInteractionCell(), capacity=3)
    a,b,c=torch.eye(8)[:3]
    store.observe('a',a,0,[1,0,0,0]);store.observe('b',b,1,[0,1,0,0])
    before=store.snapshot()
    assert store.memory.lookup(b)['entity_id']==1
    assert store.memory.lookup(c)['entity_id'] is None
    assert store.snapshot()==before
    with pytest.raises(LookupError):
        store.observe('unknown',a,2,[0,0,0,0],source_query=c)
    assert store.snapshot()==before


def test_query_payload_replay_and_snapshot_binding():
    matcher=Scorer();cell=EntityInteractionCell()
    store=EntityStateMemory(matcher,cell,capacity=3)
    a,b,c=torch.eye(8)[:3]
    store.observe('a',a,0,[1,0,0,0]);store.observe('b',b,1,[0,1,0,0])
    before=copy.deepcopy(store.latents)
    receipt=store.observe('copy',a,2,[0,0,0,0],source_query=b)
    assert receipt['source_id']==1 and store.latents[1]==before[1]
    store.observe('c',c,3,[1,0,0,0])
    snapshot=store.snapshot()
    assert store.observe('copy',a,2,[0,0,0,0],source_query=b)==receipt
    assert store.snapshot()==snapshot
    changed=torch.nn.functional.normalize(b+.01*c,dim=0)
    assert store.memory.lookup(changed)['entity_id']==1
    with pytest.raises(ValueError,match='Conflicting'):
        store.observe('copy',a,2,[0,0,0,0],source_query=changed)
    assert store.snapshot()==snapshot
    assert EntityStateMemory.restore(matcher,cell,snapshot).snapshot()==snapshot
    bad=copy.deepcopy(snapshot);bad['memory']['receipts'][2]['result']['source_id']=99
    with pytest.raises(ValueError):
        EntityStateMemory.restore(matcher,cell,bad)


def test_three_record_sequence_preserves_other_states():
    cell=EntityInteractionCell()
    observations=torch.tensor([[[1.,0,0,0],[0,1.,0,0],[1.,0,0,0],[0,0,0,0]]])
    slots=torch.tensor([[0,1,2,2]])
    sources=torch.tensor([[-1,-1,-1,1]])
    _,before=cell(observations[:,:3],slots[:,:3],sources[:,:3],entity_count=3)
    _,after=cell(observations,slots,sources,entity_count=3)
    assert torch.equal(before[:,:2],after[:,:2])
    assert torch.allclose(after[:,2],cell.interact(before[:,2],before[:,1]))
