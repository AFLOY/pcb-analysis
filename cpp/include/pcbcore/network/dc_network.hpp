// Direct DC solve of a conductance network: the zero-frequency sheet mesh.
//
// The caller numbers the nodes and lists the branches ``left[k] -- right[k]``
// with conductances in siemens; node ``reference`` is held at 0 V and removed.
// The reduced Laplacian is factored once (SuperLU) and the forward injection
// and every objective's adjoint right-hand side are solved together.  Every
// accumulation runs in branch order, so a network gives the same bits on
// every call and at every thread budget.
#pragma once

#include <cstdint>
#include <vector>

#include "pcbcore/linalg/sparse_lu.hpp"

namespace pcbcore::network {

struct ConductanceNetworkView {
    std::int64_t node_count{0};
    std::int64_t branch_count{0};
    const std::int64_t* left{nullptr};
    const std::int64_t* right{nullptr};
    const double* conductance{nullptr};

    // Throws InvalidInput when an endpoint lies outside 0..node_count-1.
    void validate() const;
};

// The Laplacian with the reference row and column removed.  Unknowns keep
// the node order with the reference left out.
[[nodiscard]] linalg::CscMatrix reduced_laplacian(const ConductanceNetworkView& network,
                                                  std::int64_t reference);

struct DCNetworkResult {
    std::vector<double> voltage_unknowns;  // node_count - 1
    std::vector<double> node_voltage;      // node_count, reference at 0 V
    std::vector<double> edge_current;      // branch_count, left -> right
    std::vector<double> node_current;      // node_count, net current out of each node
    std::vector<double> adjoint_voltage;   // objectives x node_count, row-major
    double loss_w{0.0};
    double relative_residual{0.0};
    bool singular{false};
};

// ``injection`` holds node_count currents entering the nodes; ``weights`` is
// objectives x node_count (row-major) or null with objectives == 0.  A
// singular Laplacian gives NaN potentials, not an exception.
[[nodiscard]] DCNetworkResult solve_conductance_network(const ConductanceNetworkView& network,
                                                        std::int64_t reference,
                                                        const double* injection,
                                                        const double* weights,
                                                        std::int64_t objectives);

// ``conductance * branch_product`` per branch, half of each in-plane branch
// credited to each endpoint, and the vertical (not in-plane) total.
struct BranchSensitivity {
    std::vector<double> node_sensitivity;
    std::vector<double> branch_sensitivity;
    double vertical_total{0.0};
};

[[nodiscard]] BranchSensitivity split_branch_sensitivity(const ConductanceNetworkView& network,
                                                         const double* branch_product,
                                                         const bool* in_plane);

}  // namespace pcbcore::network
