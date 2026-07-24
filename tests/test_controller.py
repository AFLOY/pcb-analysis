import unittest

from peec_fastopt.backends import SyntheticBackend
from peec_fastopt.controller import (
    CandidateEstimate,
    DynamicController,
    ExecutionReport,
    FidelityStage,
    GIB,
    ProblemProfile,
)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.problem = ProblemProfile(
            nx=1024, ny=1024, layers=4, unknowns=2_000_000,
            candidate_count=1000, changed_cells_mean=32,
        )

    def test_4gb_and_8gb_plans_fit_safe_budgets(self):
        controller = DynamicController()
        plan4 = controller.make_plan(
            self.problem, SyntheticBackend(4 * GIB).probe(), FidelityStage.CORRECTION
        )
        plan8 = controller.make_plan(
            self.problem, SyntheticBackend(8 * GIB).probe(), FidelityStage.CORRECTION
        )
        self.assertLessEqual(plan4.estimated_bytes, plan4.memory_budget_bytes)
        self.assertLessEqual(plan8.estimated_bytes, plan8.memory_budget_bytes)
        self.assertGreaterEqual(plan8.tile_size, plan4.tile_size)

    def test_oom_replan_reduces_tile(self):
        controller = DynamicController()
        telemetry = SyntheticBackend(4 * GIB).probe()
        initial = controller.make_plan(self.problem, telemetry, FidelityStage.CORRECTION)
        oom = ExecutionReport(0, 0.0, converged=False, oom=True)
        retry = controller.make_plan(
            self.problem, telemetry, FidelityStage.CORRECTION,
            previous_report=oom, previous_plan=initial,
        )
        self.assertLess(retry.tile_size, initial.tile_size)

    def test_stagnation_switches_bicgstab_to_gmres(self):
        controller = DynamicController()
        telemetry = SyntheticBackend(8 * GIB).probe()
        first = controller.make_plan(self.problem, telemetry, FidelityStage.CORRECTION)
        self.assertEqual(first.solver, "bicgstab2")
        stalled = ExecutionReport(
            peak_bytes=first.estimated_bytes, elapsed_ms=10.0,
            converged=False, stagnated=True,
        )
        retry = controller.make_plan(
            self.problem, telemetry, FidelityStage.CORRECTION,
            previous_report=stalled, previous_plan=first,
        )
        self.assertEqual(retry.solver, "gmres")
        self.assertGreaterEqual(retry.restart, 12)

    def test_error_intervals_and_high_risk_trigger_promotion(self):
        controller = DynamicController()
        candidates = [
            CandidateEstimate("best", 1.000, split_error=0.001),
            CandidateEstimate("overlap", 1.001, split_error=0.001),
            CandidateEstimate("clear", 1.100, split_error=0.0001),
            CandidateEstimate("via", 1.200, high_risk_topology=True),
            CandidateEstimate("other", 1.300),
        ]
        promoted = controller.select_for_promotion(candidates)
        self.assertIn("best", promoted)
        self.assertIn("overlap", promoted)
        self.assertIn("via", promoted)
        self.assertNotIn("clear", promoted)

    def test_observation_updates_memory_scale(self):
        controller = DynamicController()
        telemetry = SyntheticBackend(8 * GIB).probe()
        plan = controller.make_plan(self.problem, telemetry, FidelityStage.NEAR_FINE)
        before = controller.runtime.memory_scale
        controller.observe(plan, ExecutionReport(
            peak_bytes=int(plan.estimated_bytes * 1.4), elapsed_ms=4.0
        ))
        self.assertGreater(controller.runtime.memory_scale, before)

    def test_partial_cuda_calibration_does_not_corrupt_full_memory_model(self):
        controller = DynamicController()
        telemetry = SyntheticBackend(8 * GIB).probe()
        plan = controller.make_plan(self.problem, telemetry, FidelityStage.NEAR_FINE)
        before = controller.runtime.memory_scale
        controller.observe(plan, ExecutionReport(
            peak_bytes=1, elapsed_ms=4.0, memory_complete=False
        ))
        self.assertEqual(controller.runtime.memory_scale, before)

    def test_missed_audit_tightens_runtime_policy(self):
        controller = DynamicController()
        old_fraction = controller.policy.shortlist_fraction
        old_radius = controller.policy.coarse_near_radius
        controller.record_accuracy_audit(
            truly_promising=True, was_shortlisted=False,
            coarse_fine_relative_gap=2e-3,
        )
        self.assertGreater(controller.policy.shortlist_fraction, old_fraction)
        self.assertGreater(controller.policy.coarse_near_radius, old_radius)
        plan = controller.make_plan(
            self.problem, SyntheticBackend(8 * GIB).probe(),
            FidelityStage.NEAR_COARSE,
        )
        self.assertEqual(plan.near_radius, controller.policy.coarse_near_radius)

    def test_precision_gap_promotes_field_storage_to_complex128(self):
        controller = DynamicController()
        telemetry = SyntheticBackend(8 * GIB).probe()
        report = ExecutionReport(
            peak_bytes=1, elapsed_ms=1.0, precision_gap=2e-5
        )
        plan = controller.make_plan(
            self.problem, telemetry, FidelityStage.CORRECTION,
            previous_report=report,
        )
        self.assertEqual(plan.field_precision, "complex128")

        refined = controller.make_plan(
            self.problem,
            telemetry,
            FidelityStage.REFINED,
            previous_report=ExecutionReport(peak_bytes=1, elapsed_ms=1.0),
        )
        self.assertEqual(refined.field_precision, "complex128")


if __name__ == "__main__":
    unittest.main()
