// A board and the bodies in contact with it, each on its own thermal mesh
// (multiphysics/staggered_coupling/{board_enclosure,electro_thermal_enclosure}.py,
// thermal/matrix_free_mpir_fem/contact.py): the contact transfers, the
// Dirichlet-Neumann interface iteration, and the electro-thermal loop
// around it.
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "pcbcore/coupling/coupling.hpp"
#include "pcbcore/thermal/thermal_system.hpp"

namespace pcbcore::coupling {

// Element-face pairs across one board/body interface.
struct ContactMap {
    bool board_top{true};
    int body_face{4};                // direction of the body's face (-z on the board's top)
    std::vector<Index> board_cells;  // (pairs, 2): row, col
    std::vector<Index> body_cells;   // (pairs, 3): slab, row, col
    std::vector<double> conductance; // (pairs,), W/K

    [[nodiscard]] Index size() const noexcept { return static_cast<Index>(conductance.size()); }
    [[nodiscard]] std::vector<double> board_face_temperature(const thermal::ThermalMesh& board,
                                                             const double* temperature) const;
    [[nodiscard]] std::vector<double> body_face_temperature(const thermal::ThermalMesh& body,
                                                            const double* temperature) const;
    // The film coefficient and ambient the board sees from the body.
    [[nodiscard]] thermal::FaceBoundary board_robin(const thermal::ThermalMesh& board,
                                                    const std::vector<double>& body_temperature) const;
    // The pair heat lumped onto the four body face nodes of each pair.
    [[nodiscard]] std::vector<double> body_nodal_heat(const thermal::ThermalMesh& body,
                                                      const std::vector<double>& pair_heat) const;
};

struct Body {
    std::shared_ptr<const thermal::ThermalProblem> problem;
    ContactMap contact;
    double start{0.0};  // contact temperature before the first exchange
};

struct InterfaceConfig {
    int max_iterations{50};
    double temperature_tolerance{1.0e-4};
    double relative_heat_tolerance{1.0e-6};
    double relaxation{1.0};
    bool aitken{true};
    double max_relaxation{4.0};
    double divergence_temperature{1.0e6};
    thermal::SolveOptions board;
    thermal::SolveOptions body;
};

struct InterfaceStep {
    int iteration{0};
    double interface_heat{0.0};
    double max_temperature_change{0.0};
    double relative_heat_change{0.0};
    double relaxation{1.0};
    int board_inner_iterations{0};
    int body_inner_iterations{0};
};

struct EnclosureResult {
    thermal::ThermalSolution board;
    std::vector<thermal::ThermalSolution> bodies;
    std::vector<std::vector<double>> contact_heat;
    std::vector<std::vector<double>> contact_temperature;
    bool converged{false};
    std::vector<InterfaceStep> history;

    [[nodiscard]] double interface_heat() const;
};

// run_board_enclosure_thermal; ``initial`` (or null) is a previous result of
// the same geometry.
[[nodiscard]] EnclosureResult run_board_enclosure(const thermal::ThermalProblem& board, const std::vector<Body>& bodies,
                                                  const InterfaceConfig& config, const EnclosureResult* initial);

struct EnclosureStep {
    CouplingStep step;
    int interface_iterations{0};
    double interface_heat{0.0};
};

struct ElectroThermalEnclosureResult {
    DCSolution electrical;
    EnclosureResult thermal;
    std::vector<double> conductivity;
    std::vector<double> via_resistance;
    std::vector<double> element_temperature;
    bool converged{false};
    std::vector<EnclosureStep> history;
    double cold_loss{std::numeric_limits<double>::quiet_NaN()};
};

[[nodiscard]] ElectroThermalEnclosureResult run_electro_thermal_enclosure(
    const DCProblem& problem, const Placement& placement, const thermal::ThermalProblem& board,
    const ExtraHeat& extra, const std::vector<Body>& bodies, const FixedPointConfig& config,
    const InterfaceConfig& interface, int threads);

}  // namespace pcbcore::coupling
