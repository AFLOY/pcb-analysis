from __future__ import annotations

from experiments.peec_fem_comparison import (
    dc_strip_case,
    peec_skin_effect_case,
    skin_effect_case,
)


def test_peec_and_fem_match_the_same_dc_strip_resistance() -> None:
    result = dc_strip_case(8, 1, include_cuda=False)

    assert result["peec"]["relative_error"] < 1.0e-12
    assert result["fem"]["relative_error"] < 1.0e-10
    assert abs(
        result["peec"]["resistance_ohm"]
        - result["fem"]["resistance_ohm"]
    ) / result["exact_resistance_ohm"] < 1.0e-10


def test_fem_skin_impedance_converges_to_peec_slab_reference() -> None:
    result = skin_effect_case(64, 1)

    assert result["fem_relative_error_vs_peec_slab_reference"] < 1.3e-3
    assert result["peec_reference_relative_error_vs_closed_form"] < 1.0e-11


def test_fem_reports_the_slab_ac_to_dc_ratio_it_is_compared_on() -> None:
    result = skin_effect_case(32, 1)

    assert abs(result["closed_form_ac_to_dc_ratio"] - 3.788) < 1.0e-3
    assert result["fem_ac_to_dc_ratio_relative_error"] < 5.0e-3


def test_peec_solves_the_skin_slab_with_filaments_rather_than_a_formula() -> None:
    # A short bar, so the test stays affordable: the middle is still within the
    # terminals' reach and the ratio has not yet risen to the slab's.  What
    # must hold regardless of length is that the solve converged, that the
    # current has left the DC division (ratio one) toward the faces, and that
    # the filament currents divide the way a slab's would.
    result = peec_skin_effect_case(8, 1)

    assert result["peec_converged"]
    assert result["filaments"] == 9
    assert 1.5 < result["peec_ac_to_dc_ratio"] < result["closed_form_ac_to_dc_ratio"]
    assert result["peec_profile_relative_l2_error_vs_filament_averaged_slab"] < 0.25
    assert result["peec_residual"] < 1.0e-8
    assert "cuda_total" not in result
