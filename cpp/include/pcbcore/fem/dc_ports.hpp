// The N-port of one layered DC conductor (``ports.dc_port_basis``): n - 1
// unit voltage solves on one prepared system, the conductance matrix closed
// by KCL, and the unit-current fields that turn any port excitation back
// into a potential, plus the time-averaged loss and RMS current density of
// a port-current correlation.
#pragma once

#include <cstdint>
#include <vector>

#include "pcbcore/fem/layered_dc_system.hpp"
#include "pcbcore/fem/mpir.hpp"

namespace pcbcore::fem {

struct DCPortBasisResult {
    std::int64_t ports{0};
    std::vector<double> conductance;   // (ports, ports)
    std::vector<double> unit_voltage;  // (ports - 1, nodes): driven port at 1 V
    std::vector<double> unit_current;  // (ports - 1, nodes): 1 A in, out at the reference
    std::vector<MpirResult> solves;    // one per driven port, in port order
};

// ``ports`` are the pads (every node a Dirichlet node of ``system``);
// ``initial_unit_voltage`` (ports - 1, nodes) warm-starts the free nodes of
// each unit solve, or is null.  The unit solves run on ``width`` threads,
// each driving an OpenMP team of ``team``; the result does not depend on
// either.
[[nodiscard]] DCPortBasisResult dc_port_basis(const LayeredDCSystem& system, const NodeGroupsView& ports,
                                              std::int64_t reference, const double* initial_unit_voltage,
                                              const MpirConfig& config, int width, int team);

// The modes of a port-current correlation restricted to the driven ports:
// ``reduced`` (driven, driven) is diagonalised, eigenvalues below
// -1e-10 scale are rejected (InvalidInput), those above 1e-14 scale kept.
struct CorrelationModes {
    std::vector<double> weights;  // (kept,)
    std::vector<double> modes;    // (kept, nodes): sum_k u_mk phi_k
};
[[nodiscard]] CorrelationModes correlation_modes(const double* reduced, std::int64_t driven, double scale,
                                                 const double* unit_current, std::int64_t nodes);

struct ModalLoss {
    std::vector<double> element;  // (layers, rows, cols), W
    std::vector<double> via;      // (vias,), W
};
// sum_m w_m P(psi_m) element by element and via by via.
[[nodiscard]] ModalLoss modal_loss(const LayeredDCSystem& system, const CorrelationModes& modes, int threads);

// sqrt(sum_m w_m |sigma E(psi_m)|^2) per element.
[[nodiscard]] std::vector<double> modal_rms_current_density(const LayeredDCSystem& system,
                                                            const CorrelationModes& modes, int threads);

}  // namespace pcbcore::fem
