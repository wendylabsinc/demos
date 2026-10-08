#!/usr/bin/env python3
"""Run each regression module in an isolated process; save every test outcome."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parent
PHYSICAL=HERE.parent

def integrity():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        path=PHYSICAL/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Missing or changed captured evidence: '+relative)
    return manifest

class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.cases=[]
    def record(self,test,status,detail=None):
        self.cases.append(dict(test=test.id(),status=status,detail=detail))
    def addSuccess(self,test):super().addSuccess(test);self.record(test,'passed')
    def addFailure(self,test,err):super().addFailure(test,err);self.record(test,'failed',self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.record(test,'error',self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.record(test,'skipped',reason)
    def addExpectedFailure(self,test,err):super().addExpectedFailure(test,err);self.record(test,'expected_failure',self._exc_info_to_string(err,test))
    def addUnexpectedSuccess(self,test):super().addUnexpectedSuccess(test);self.record(test,'unexpected_success')
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        if err is not None:self.record(subtest,'failed',self._exc_info_to_string(err,test))

def child(module,output):
    sys.path.insert(0,str(PHYSICAL))
    suite=unittest.defaultTestLoader.loadTestsFromName(module)
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=1,resultclass=Result).run(suite)
    Path(output).write_text(json.dumps(dict(module=module,tests_run=result.testsRun,
        seconds=time.monotonic()-started,successful=result.wasSuccessful(),cases=result.cases),indent=2))
    return 0 if result.wasSuccessful() else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['regression','simulation','open','model','all','today'],default='regression')
    parser.add_argument('--test',help='One unittest module or dotted test ID')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--report',type=Path,default=HERE/'reports/latest.json')
    parser.add_argument('--timeout',type=int,default=180,help='Maximum seconds per module')
    parser.add_argument('--child',nargs=2,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child:return child(*args.child)
    manifest=integrity()
    simulations=['test_sim_color_depth','test_sim_runtime','test_simulation_position_noise','test_sim_position_noise']
    modules=sorted(p.stem for p in PHYSICAL.glob('test_*.py') if p.stem not in simulations)
    modules+=['offline.open_cases']
    if args.profile=='simulation':modules=list(simulations)
    elif args.profile=='open':modules=['offline.open_cases']
    elif args.profile=='model':modules=['offline.test_model_recognition']
    elif args.profile=='all':modules+=simulations
    elif args.profile=='today':
        from today import MODULES
        modules=list(MODULES)
    if args.test:modules=[args.test]
    if args.list:
        print('\n'.join(modules));return 0
    started=time.monotonic();results=[]
    simulation_dir=Path('simulations')/str(time.time_ns())
    with tempfile.TemporaryDirectory(prefix='g1-offline-') as directory:
        env={k:v for k,v in os.environ.items() if not k.startswith(('HANDOFF_','G1_'))}
        env.update(HANDOFF_OFFLINE='1',HANDOFF_DATA=directory,HANDOFF_CAMERA_MAX_AGE_MS='400',
                   PYTHONPATH=os.pathsep.join([str(HERE),str(PHYSICAL)]),
                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONHASHSEED='0',YOLO_CONFIG_DIR=directory)
        if args.profile=='today':
            env.update(HANDOFF_CAMERA_MAX_AGE_MS='500',
                       HANDOFF_SIM_ARTIFACT_DIR=str(args.report.resolve().parent/simulation_dir))
        # macOS uses its native renderer; Linux simulation requires EGL.
        if sys.platform!='darwin':env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
        for i,module in enumerate(modules):
            output=Path(directory)/('result-'+str(i)+'.json')
            print('Running '+module,flush=True)
            try:
                process=subprocess.run([sys.executable,str(__file__),'--child',module,str(output)],
                    cwd=PHYSICAL,env=env,timeout=args.timeout,capture_output=True,text=True)
                if process.stdout:print(process.stdout,end='')
                if process.stderr:print(process.stderr,end='',file=sys.stderr)
                result=json.loads(output.read_text()) if output.exists() else dict(module=module,tests_run=0,cases=[],successful=False)
                result.update(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr)
                if process.returncode!=0:result['successful']=False
            except subprocess.TimeoutExpired:
                result=dict(module=module,tests_run=0,cases=[],successful=False,error='Module timeout')
            results.append(result)
    total=sum(r['tests_run'] for r in results);passed=all(r['successful'] for r in results)
    report=dict(profile=args.profile,tests_run=total,successful=passed,seconds=time.monotonic()-started,
        evidence_verified=len(manifest['files']),hardware_used=False,physical_success_verified=False,
        unresolved=['live_camera_headroom','live_control_handover','physical_pickup'],modules=results)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    if args.profile=='today':
        from today import dashboard
        report['simulation_artifact_dir']=simulation_dir.as_posix()
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        dashboard(report,args.report.with_name('index.html'),args.report.name)
    print(f'{total} tests; '+('PASS' if passed else 'FAIL')+f'; report: {args.report}')
    print('Live camera timing, ownership settling, and a physical pickup remain unverified.')
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
