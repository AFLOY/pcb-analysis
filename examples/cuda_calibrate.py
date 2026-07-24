"""Run controller/FFT calibration on a machine with CUDA and CuPy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from peec_fastopt.controller import DynamicController, FidelityStage, ProblemProfile
from peec_fastopt.cupy_calibrator import CuPyCalibrationBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--unknowns", type=int, default=2_000_000)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--changed-cells", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    backend = CuPyCalibrationBackend(repeats=args.repeats)
    telemetry = backend.probe()
    problem = ProblemProfile(
        nx=args.grid, ny=args.grid, layers=args.layers,
        unknowns=args.unknowns, candidate_count=args.candidates,
        changed_cells_mean=args.changed_cells,
    )
    controller = DynamicController()
    output = {
        "telemetry": asdict(telemetry),
        "stages": [],
    }
    for stage in (
        FidelityStage.NEAR_COARSE,
        FidelityStage.NEAR_FINE,
        FidelityStage.CORRECTION,
        FidelityStage.REFINED,
    ):
        plan = controller.make_plan(problem, telemetry, stage)
        report = backend.execute(plan, problem)
        if report.oom:
            plan = controller.make_plan(
                problem, backend.probe(), stage,
                previous_report=report, previous_plan=plan,
            )
            report = backend.execute(plan, problem)
        controller.observe(plan, report)
        output["stages"].append({
            "stage": stage.name,
            "plan": asdict(plan),
            "fft_calibration": asdict(report),
        })
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
