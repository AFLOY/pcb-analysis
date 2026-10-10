// MPIR configuration and results between Python and the C++ core.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>
#include <vector>

#include "pcbcore/fem/mpir.hpp"

namespace pcbcore_bindings {

namespace py = pybind11;

inline pcbcore::fem::MpirConfig mpir_config(const double relative_tolerance, const double absolute_tolerance,
                                     const double inner_relative_tolerance, const int max_outer_iterations,
                                     const int max_inner_iterations) {
    pcbcore::fem::MpirConfig config;
    config.relative_tolerance = relative_tolerance;
    config.absolute_tolerance = absolute_tolerance;
    config.inner_relative_tolerance = inner_relative_tolerance;
    config.max_outer_iterations = max_outer_iterations;
    config.max_inner_iterations = max_inner_iterations;
    return config;
}

inline py::dict result_dict(const pcbcore::fem::MpirResult& result) {
    std::vector<std::tuple<int, double, int, double>> history;
    for (const auto& step : result.history) {
        history.emplace_back(step.outer_iteration, step.high_relative_residual, step.inner_iterations,
                             step.inner_relative_residual);
    }
    py::dict out;
    out["converged"] = result.converged;
    out["outer_iterations"] = result.outer_iterations;
    out["inner_iterations"] = result.inner_iterations;
    out["relative_residual"] = result.relative_residual;
    out["high_operator_applications"] = result.high_operator_applications;
    out["low_operator_applications"] = result.low_operator_applications;
    out["history"] = history;
    return out;
}

}  // namespace pcbcore_bindings
