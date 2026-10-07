"""User requests are constrained to the catalog and supersede earlier missions."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from items import NAMES, requested_item
from fastapi.testclient import TestClient
import demo

@pytest.mark.parametrize('name',NAMES)
def test_natural_request(name):
    assert requested_item(f'Please find the {name}.') == name

@pytest.mark.parametrize('query',['find my keys','find the bottle and chair','find something',''])
def test_invalid_request(query):
    with pytest.raises(ValueError): requested_item(query)

def test_new_request_cancels_previous_destination():
    client=TestClient(demo.app)
    first=client.post('/api/search',json={'query':'find the apple'}).json()
    demo.shared.update(destination=[3,2],route=[[2,1]],camera='cam2',confidence=.9)
    second=client.post('/api/search',json={'query':'Find the suitcase'}).json()
    status=client.get('/api/status').json()
    assert second['request_id']>first['request_id']
    assert status['requested_item']=='suitcase'
    assert status['destination'] is None and status['route']==[]
    assert status['camera'] is None and status['confidence'] is None
    assert demo.commands['search']=='suitcase'
    assert client.post('/api/search',json={'query':'apple or banana'}).status_code==422
    assert demo.commands['search']=='suitcase'

def test_twin_route_avoids_intervening_props():
    import numpy as np
    demo.make_scene(objects=True)
    start=(1,-3.75)
    goal,route=demo.approach(start,(4.2,-3.2))
    assert demo.blocked(2.4,-3.6)  # Bowl blocks the previous direct route.
    points=[start]+route
    for a,b in zip(points,points[1:]):
        for p in np.linspace(a,b,30):
            assert not demo.blocked(*p),p
