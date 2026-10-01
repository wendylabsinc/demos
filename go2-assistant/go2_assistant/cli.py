"""Run the Wendy Go2 assistant in simulation or against commissioned hardware."""
import argparse
import json
import os
from pathlib import Path


def build_assistant(args):
    from .core import Assistant
    from .memory import Memory
    if args.backend == 'hardware':
        from .hardware import HardwareBackend
        if not args.observation_url or not args.map_id:
            raise ValueError('Hardware needs --observation-url and a stable --map-id.')
        backend = HardwareBackend(args.observation_url, args.map_id, args.navigator_url, args.motion_url)
        memory = Memory(args.memory, args.map_id)
        if args.arm:
            result = backend.arm()
            if not result['accepted']:
                raise ValueError(result['reason'])
    else:
        if args.arm:
            raise ValueError('--arm is only for commissioned hardware.')
        from .sim import SimBackend
        mapping_path = args.mapping or str(Path(args.memory).with_suffix('.map.npz'))
        backend = SimBackend(args.assets, mapping_path=mapping_path)
        memory = Memory(args.memory, backend.map_id)
    return Assistant(backend, memory)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['serve', 'demo', 'evaluate'], nargs='?', default='serve')
    p.add_argument('--backend', choices=['sim', 'hardware'], default=os.getenv('GO2_BACKEND', 'sim'))
    p.add_argument('--memory', default=os.getenv('GO2_MEMORY', 'results/memory.sqlite3'))
    p.add_argument('--assets', default=os.getenv('GO2_ASSETS'))
    p.add_argument('--mapping', default=os.getenv('GO2_MAPPING'), help='Persistent sensor-built occupancy map')
    p.add_argument('--port', type=int, default=int(os.getenv('PORT', '8787')))
    p.add_argument('--host', default=os.getenv('GO2_HOST', '127.0.0.1'))
    p.add_argument('--ssl-certfile')
    p.add_argument('--ssl-keyfile')
    p.add_argument('--record', action='store_true')
    p.add_argument('--output', default='results/demo')
    p.add_argument('--observation-url', default=os.getenv('GO2_OBSERVATION_URL'))
    p.add_argument('--map-id', default=os.getenv('GO2_MAP_ID'))
    p.add_argument('--navigator-url', default=os.getenv('GO2_NAVIGATOR_URL', 'http://127.0.0.1:3202'))
    p.add_argument('--motion-url', default=os.getenv('GO2_MOTION_URL', 'http://127.0.0.1:3201'))
    p.add_argument('--arm', action='store_true', help='Operator arms commissioned hardware; still waits for a task command')
    return p


def validate_args(args, p):
    if not 1 <= args.port <= 65535:
        p.error('Port must be between 1 and 65535.')
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        p.error('HTTPS needs both --ssl-certfile and --ssl-keyfile.')
    if args.backend == 'hardware' and args.mode != 'serve':
        p.error('Demo and evaluation are simulator-only.')
    if args.mode == 'serve' and args.host not in {'127.0.0.1', 'localhost', '::1'} and not os.getenv('GO2_HTTP_TOKEN'):
        p.error('Set GO2_HTTP_TOKEN before exposing the phone server beyond localhost.')


def main():
    p = parser()
    args = p.parse_args()
    validate_args(args, p)
    try:
        if args.mode == 'evaluate':
            from .evaluation import evaluate
            report = evaluate(args.assets, Path(args.output))
            print(json.dumps(report, indent=2))
            return 0 if report['passed'] else 1
        assistant = build_assistant(args)
        try:
            if args.mode == 'demo':
                from .evaluation import demo
                report = demo(assistant, Path(args.output), record=args.record)
                print(json.dumps(report, indent=2))
                return 0 if report['passed'] else 1
            import uvicorn
            from .server import create_app
            print(f"Wendy Go2 phone interface on port {args.port}", flush=True)
            uvicorn.run(create_app(assistant), host=args.host, port=args.port,
                        ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
            return 0
        finally:
            assistant.stop()
            if hasattr(assistant.backend, 'close'):
                assistant.backend.close()
            assistant.memory.close()
    except (ValueError, RuntimeError, OSError) as error:
        p.exit(1, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
