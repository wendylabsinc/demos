"""Command line entry points for the Wendy vacuum simulator."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run a robot vacuum in a virtual home.")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Open the interactive 3D home and cleaning controls")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18881)
    serve.add_argument("--home", type=Path)
    serve.add_argument("--seed", type=int, default=7)
    record = sub.add_parser("record", help="Run a complete mission and render an MP4")
    record.add_argument("--output", type=Path, default=Path("media/maple-house.mp4"))
    record.add_argument("--home", type=Path)
    record.add_argument("--seed", type=int, default=7)
    record.add_argument("--speed", type=float, default=20, help="Simulated seconds per video second")
    record.add_argument("--fps", type=int, default=24)
    record.add_argument("--width", type=int, default=1280)
    record.add_argument("--height", type=int, default=720)
    record.add_argument("--max-time", type=float, default=1800, help="Mission limit in simulated seconds")
    record.add_argument("--camera", choices=["tour", "overview", "top"], default="tour")
    args = parser.parse_args()
    if args.command == "record":
        from .record import render_mission
        render_mission(args)
    else:
        from .server import run_server
        run_server(getattr(args, "host", "127.0.0.1"), getattr(args, "port", 18881),
                   getattr(args, "home", None), getattr(args, "seed", 7))


if __name__ == "__main__":
    main()
