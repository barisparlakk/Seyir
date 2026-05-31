"""
evaluate.py — Batch evaluation across all scenarios and seeds.

Runs each scenario N times with different seeds and aggregates metrics
into a summary report saved to logs/evaluation_report.json.

Usage:
    python scripts/evaluate.py --runs 3 --duration 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("evaluate")

METRICS_KEYS = [
    "collision_count",
    "traffic_violation_count",
    "avg_speed_kmh",
    "route_completion_pct",
    "min_ttc",
    "avg_longitudinal_jerk",
    "emergency_brake_count",
    "runtime_seconds",
]


def load_run_metrics(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("summary", {})
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def aggregate(metrics_list: list[dict]) -> dict:
    """Compute mean ± std across multiple runs."""
    result: dict = {}
    for key in METRICS_KEYS:
        vals = [m[key] for m in metrics_list if key in m]
        if not vals:
            result[key] = {"mean": None, "std": None}
        else:
            result[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
            }
    return result


def run_evaluation(args: argparse.Namespace) -> None:
    import subprocess

    scenarios = ["narrow_street", "unmarked_intersection", "village_road"]
    report: dict = {}

    for scenario in scenarios:
        print(f"\n── Evaluating scenario: {scenario} ({args.runs} runs) ──")
        run_metrics: list[dict] = []

        for seed in range(args.seed_start, args.seed_start + args.runs):
            run_id = f"{scenario}_s{seed}"
            out_path = LOG_DIR / f"metrics_{run_id}.json"

            print(f"  seed={seed} → {out_path.name}")
            cmd = [
                sys.executable, "scripts/run_simulation.py",
                "--scenario",  scenario,
                "--seed",      str(seed),
                "--duration",  str(args.duration),
                "--record",
                "--no-render",
                "--host",      args.host,
                "--port",      str(args.port),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    timeout=args.duration + 60,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.warning("Run failed (seed=%d): %s", seed, result.stderr[-500:])
                    continue
            except subprocess.TimeoutExpired:
                logger.warning("Run timed out (seed=%d)", seed)
                continue

            m = load_run_metrics(out_path)
            if m:
                run_metrics.append(m)
                print(f"    collisions={m['collision_count']}  "
                      f"completion={m['route_completion_pct']:.1f}%  "
                      f"avg_speed={m['avg_speed_kmh']:.1f} km/h")

        report[scenario] = aggregate(run_metrics) if run_metrics else {}

    # Print summary table
    print("\n═══ Evaluation Summary ═══════════════════════════════════")
    for scenario, agg in report.items():
        print(f"\n{scenario.upper()}")
        for key in METRICS_KEYS:
            stats = agg.get(key, {})
            if stats.get("mean") is not None:
                print(f"  {key:<30}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    # Save full report
    report_path = LOG_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-evaluate Seyir across all scenarios")
    parser.add_argument("--runs",       type=int, default=3,
                        help="Number of runs per scenario")
    parser.add_argument("--duration",   type=int, default=60,
                        help="Duration per run in seconds")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--host",       default="localhost")
    parser.add_argument("--port",       type=int, default=2000)
    run_evaluation(parser.parse_args())
