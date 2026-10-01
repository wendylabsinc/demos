"""Run in Wendy Sandbox: PYTHONPATH=/opt/sandbox/sim uv run wendy-simrun wendy_demo.py."""
import uvicorn
import wendymujoco
from go2_assistant.cli import build_assistant, parser, validate_args
from go2_assistant.server import create_app

p = parser()
args = p.parse_args(['serve'])
validate_args(args, p)
if args.backend != 'sim':
    p.error('The Wendy Sim tab entrypoint requires simulator mode.')
assistant = build_assistant(args)
backend = assistant.backend
# MuJoCo and offscreen graphics share the main thread with Uvicorn. The Sim
# bridge writes geometry and poses; Wendy renders those in its own Sim tab.
with wendymujoco.launch_passive(backend.sim.model, backend.sim.data,
                              title='Wendy Go2 home assistant') as viewer:
    original_tick = assistant.tick
    def tick_and_publish(dt=.1):
        original_tick(dt)
        with assistant.lock:
            status = assistant.status()
            viewer.hud(task=status['task']['kind'] if status['task'] else 'ready',
                       state=status['backend']['state'], map=status['map_id'],
                       phone_port=args.port)
            viewer.sync()
    assistant.tick = tick_and_publish
    print(f'Phone chat on port {args.port}. Start with: explore the house or find a cup. '
          'Room names start empty; teach one with: remember this as kitchen. '
          'Default YOLOS recognizes fixed object classes; keys, colors and ownership are unsupported.', flush=True)
    try:
        uvicorn.run(create_app(assistant), host=args.host, port=args.port, log_level='info')
    finally:
        assistant.stop()
        backend.close()
        assistant.memory.close()
