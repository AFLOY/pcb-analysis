"""Show plans selected for an emulated 4 GB or 8 GB CUDA environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .backends import SyntheticBackend
from .controller import DynamicController, FidelityStage, GIB, ProblemProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram-gb", type=float, choices=(4.0, 8.0), default=4.0)
    parser.add_argument("--platform", choices=("windows", "linux"), default="windows")
    parser.add_argument("--grid", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--unknowns", type=int, default=2_000_000)
    args = parser.parse_args()

    backend = SyntheticBackend(int(args.vram_gb * GIB), platform=args.platform)
    telemetry = backend.probe()
    problem = ProblemProfile(
        nx=args.grid, ny=args.grid, layers=args.layers,
        unknowns=args.unknowns, candidate_count=1000,
        changed_cells_mean=32,
    )
    controller = DynamicController()
    output = []
    for stage in FidelityStage:
        plan = controller.make_plan(problem, telemetry, stage)
        report = backend.execute(plan, problem)
        controller.observe(plan, report)
        row = asdict(plan)
        row["stage"] = stage.name
        row["estimated_mib"] = round(plan.estimated_bytes / 1024**2, 1)
        row["budget_mib"] = round(plan.memory_budget_bytes / 1024**2, 1)
        row["simulated_peak_mib"] = round(report.peak_bytes / 1024**2, 1)
        output.append(row)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

