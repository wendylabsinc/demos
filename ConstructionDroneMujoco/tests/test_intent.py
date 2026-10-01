"""Opt-in checks against the actual pretrained local language model."""
import os
import pytest
from yard_drone.intent import IntentService

@pytest.mark.skipif(os.environ.get('YARD_DRONE_TEST_LLM')!='1',reason='Requires local pinned Qwen server')
def test_free_form_model_interpretation():
    service=IntentService();service.ready=True
    for text,action,zone in [
        ('Please have a look at the fuel storage area','goto','B'),
        ('Stay with that worker and keep a safe distance','follow','none'),
        ('Stop flying and stay where you are','hover','none'),
        ('Record the first mission','baseline','none'),
        ('Look around for things that changed','survey','none'),
        ('There is a fire near the materials. Go investigate.','goto','A')]:
        result=service.interpret(text)
        assert len(result)==1
        assert result[0]['action']==action
        if action=='goto':assert result[0]['zone']==zone
    plan=service.interpret('Go to zone A and then land')
    assert [p['action'] for p in plan]==['goto','land']
