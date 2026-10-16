import runpy
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / 'scripts' / 'deploy.py'


def deploy(monkeypatch, *args):
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(sys, 'argv', ['deploy.py', '--device', 'jetson.local', *args])
    runpy.run_path(str(DEPLOY), run_name='__main__')
    return commands[0]


def test_builder_defaults_to_docker(monkeypatch):
    command = deploy(monkeypatch, '--profile', 'air')
    assert command[command.index('--builder') + 1] == 'docker'


@pytest.mark.parametrize('builder', ['apple-container', 'buildkit', 'docker'])
def test_builder_option_is_passed_to_wendy(monkeypatch, builder):
    command = deploy(monkeypatch, '--profile', 'air', '--builder', builder)
    assert command[command.index('--builder') + 1] == builder
    assert command.count('--builder') == 1
    assert command[:3] == ['wendy', '--json', '--device']


def test_unknown_builder_is_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        deploy(monkeypatch, '--profile', 'air', '--builder', 'podman')
