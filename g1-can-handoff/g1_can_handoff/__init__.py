"""
g1-can-handoff - A Wendy Edge Application
"""

import signal
import sys


def _signal_handler(sig, frame):
    print("Shutting down gracefully...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    from .cli import main as run

    run()


if __name__ == "__main__":
    main()
