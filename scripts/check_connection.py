"""
check_connection.py — Verify CARLA server is reachable and responsive.

Usage:
    python scripts/check_connection.py
    python scripts/check_connection.py --host 1.2.3.4 --port 2000
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def carla_handshake(host: str, port: int) -> dict:
    """Attempt a real CARLA client handshake and return server info."""
    import carla

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    version = client.get_server_version()
    world   = client.get_world()
    maps    = client.get_available_maps()
    return {
        "server_version": version,
        "current_map":    world.get_map().name,
        "available_maps": sorted(maps),
        "actors":         len(list(world.get_actors())),
    }


def main(args: argparse.Namespace) -> None:
    host, port = args.host, args.port
    print(f"\nChecking CARLA server at {host}:{port} …\n")

    # Step 1: TCP reachability
    print(f"  [1/3] TCP port {port} …", end=" ", flush=True)
    if not tcp_reachable(host, port):
        print("UNREACHABLE")
        print(f"\n  Cannot connect to {host}:{port}")
        print("  Possible causes:")
        print("    • CARLA server not started")
        print("    • Firewall blocking port 2000")
        print("    • Wrong --host address")
        print("\n  Quick-start on the remote Linux box:")
        print("    docker compose up -d carla-server")
        print("    # or without Docker:")
        print("    ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 -fps=20 &")
        sys.exit(1)
    print("OK")

    # Step 2: CARLA Python client
    print("  [2/3] CARLA Python package …", end=" ", flush=True)
    try:
        import carla
        print(f"OK (carla {carla.__version__ if hasattr(carla, '__version__') else 'installed'})")
    except ImportError:
        print("MISSING")
        print("\n  Install with:  pip install carla==0.9.15")
        sys.exit(1)

    # Step 3: Full handshake
    print("  [3/3] CARLA handshake …", end=" ", flush=True)
    try:
        info = carla_handshake(host, port)
        print("OK")
    except Exception as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    print(f"""
  ✓ Connected successfully

  Server version  : {info['server_version']}
  Current map     : {info['current_map']}
  Active actors   : {info['actors']}
  Available maps  : {", ".join(info['available_maps'][:6])}{"…" if len(info['available_maps']) > 6 else ""}

  Run a scenario from your Mac:
    python scripts/run_simulation.py --host {host} --port {port} --scenario narrow_street
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check CARLA server connectivity")
    parser.add_argument("--host", default="localhost",
                        help="IP or hostname of the CARLA server")
    parser.add_argument("--port", type=int, default=2000)
    main(parser.parse_args())
