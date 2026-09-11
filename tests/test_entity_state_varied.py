from collections import Counter
from pathwm.data.entity_state import mixed_state_episodes
from pathwm.evaluation.entity_growth import growth_inputs
from tests.test_entity_memory import Scorer


def test_complement_balance_and_simulator():
    data=mixed_state_episodes(Scorer(),growth_inputs(201,2),seed=211)
    assert data['observations'].shape==(32,12,4)
    for start in range(0,32,4):
        assert len({tuple(e['target']) for e in data['manifest'][start:start+4]})==4
    for e in data['manifest']:
        truth=[None,None]
        for entity,operation in zip(e['truth_entities'],e['operations']):
            if operation<2: truth[entity]=operation
            elif operation==2: truth[entity]^=1
        assert truth==e['target']
    assert Counter(map(tuple,data['targets'].tolist()))=={(0,0):8,(0,1):8,(1,0):8,(1,1):8}
