// Thermal solutions from the C++ core to Python dictionaries.
#pragma once

#include <pybind11/pybind11.h>

#include <utility>
#include <vector>

#include "arrays.hpp"
#include "mpir_convert.hpp"
#include "pcbcore/thermal/thermal_system.hpp"

namespace pcbcore_bindings {

using pcbcore::thermal::ThermalSolution;

inline std::vector<py::ssize_t> node_shape(const pcbcore::thermal::ThermalMesh& m) {
    return {m.slabs + 1, m.rows + 1, m.cols + 1};
}

inline py::dict solution_dict(ThermalSolution&& s, const pcbcore::thermal::ThermalMesh& mesh) {
    py::dict out;
    out["temperature"] = to_array(std::move(s.temperature), node_shape(mesh));
    out["heat_flux"] = to_array(std::move(s.heat_flux), {mesh.slabs, mesh.rows, mesh.cols, 3});
    out["max_temperature"] = s.max_temperature;
    out["min_temperature"] = s.min_temperature;
    out["total_heat_input"] = s.total_heat_input;
    out["convective_heat"] = to_array(std::move(s.convective_heat));
    out["radiative_heat"] = to_array(std::move(s.radiative_heat));
    out["fixed_temperature_heat"] = s.fixed_temperature_heat;
    out["heat_balance_error"] = s.heat_balance_error;
    out["stored_heat"] = s.stored_heat;
    py::dict solve = result_dict(s.solve);
    solve["solution"] = to_array(std::move(s.rise));
    out["solve"] = solve;
    out["radiation_iterations"] = s.radiation_iterations;
    out["radiation_converged"] = s.radiation_converged;
    out["radiation_change"] = s.radiation_change;
    return out;
}

}  // namespace pcbcore_bindings
