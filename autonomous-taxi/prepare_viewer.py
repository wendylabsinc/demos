"""Prepare a separate local Wendy viewer without altering an installed app."""
import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app',type=Path,help='Existing Wendy.app bundle')
    parser.add_argument('--port',type=int,default=8788)
    args=parser.parse_args()
    if not 1<=args.port<=65535: parser.error('Port must be between 1 and 65535')
    source=args.app.resolve()
    if not (source/'Contents/MacOS/Wendy').is_file(): parser.error('Expected a built Wendy.app')
    target=Path(__file__).resolve().parent/'runtime/Wendy Taxi.app'
    if target.exists(): parser.error(f'{target} already exists. Reuse that viewer or move it before preparing another.')
    target.parent.mkdir(exist_ok=True)
    shutil.copytree(source,target,symlinks=True)
    plist=target/'Contents/Info.plist'
    with plist.open('rb') as f: info=plistlib.load(f)
    info['CFBundleIdentifier']='sh.wendy.sandbox.taxi'
    info['CFBundleName']='Wendy Taxi'
    info.setdefault('LSEnvironment',{})['WENDY_LOCAL_SIM_PORT']=str(args.port)
    with plist.open('wb') as f: plistlib.dump(info,f)
    subprocess.run(['codesign','--force','--deep','--sign','-',str(target)],check=True)
    print(f'Open {target} after starting run.sh on port {args.port}.')

if __name__=='__main__': main()
