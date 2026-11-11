#!/usr/bin/env python3
"""Redeploy this checkout's NX2 controller and verify PASSIVE startup.

Run from any directory. --check performs only the read-only ownership check.
This does not release joints or start a demo. The controls page is local to the
Mac; UI-only edits require restarting operator_panel.py, not this deployment.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[2]
WRAPPER=WORKSPACE/'scripts/wendy-run-timed'
AUDIT=WORKSPACE/'scripts/check-wendy-deployment-wrappers'
MARKER=Path('/tmp/wendy-g1-nx2-deployment.lock')
DEVICE='10.10.20.238:50052'
VENV=WORKSPACE/'demos-g1-nx2-yolo/.venv/bin/python'

PREFLIGHT='''
import json,sys
from operator_panel import Wendy,DEVICE
assert DEVICE==sys.argv[1], 'Operator and deployment targets differ'
client=Wendy()
try:
    state=client.request(action='status')
    if state.get('adapter_initialized') is not False or state.get('demo_running') is not False:
        raise RuntimeError('Controller owns the arms or a demo is running. Support and release before redeploying.')
    if state.get('phase') not in ('PASSIVE','BLOCKED') or state.get('telemetry_healthy') is not True:
        raise RuntimeError('Idle controller state or healthy robot feedback could not be verified: '+json.dumps(state))
    print('Read-only preflight passed: controller idle, arm ownership released.')
finally:
    client.process.terminate()
'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Read-only check; do not deploy')
    parser.add_argument('--python',default=str(VENV if VENV.exists() else Path(sys.executable)),
                        help='Python with the physical project dependencies installed')
    args=parser.parse_args(argv)
    if args.check:
        subprocess.run([args.python,'-c',PREFLIGHT,DEVICE],cwd=ROOT,check=True)
        return 0
    if not WRAPPER.is_file() or not AUDIT.is_file():
        parser.error('Wendy workspace timing wrapper or deployment audit is missing: '+str(WORKSPACE/'scripts'))
    try:
        with MARKER.open('x') as marker:marker.write(f'redeploy pid={os.getpid()}\n')
    except FileExistsError:
        parser.error('Another deployment is in progress: '+str(MARKER))
    try:
        subprocess.run([sys.executable,str(AUDIT),'--root',str(WORKSPACE)],check=True)
        subprocess.run([args.python,'-c',PREFLIGHT,DEVICE],cwd=ROOT,check=True)
        env=os.environ.copy()
        env['WENDY_TIMING_FUNCTIONAL_READINESS_COMMAND']=shlex.join([args.python,str(ROOT/'verify_passive_deployment.py')])
        subprocess.run([sys.executable,str(WRAPPER),'--device',DEVICE,'--detach'],cwd=ROOT,env=env,check=True)
        print('NX2 controller redeployed and PASSIVE startup verified. No demo started.')
    finally:
        MARKER.unlink(missing_ok=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as exc:raise SystemExit(exc.returncode)
