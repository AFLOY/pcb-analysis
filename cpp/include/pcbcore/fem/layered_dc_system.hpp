// The prepared layered-PCB DC conduction system: everything one mesh, its
// vias and its fixed nodes determine, built once and reused for every solve
// on that mesh (one solve per operating point, or one per port of an N-port
// basis).
//
// It owns the per-element sheet coefficients (``c_x = σ t h_y / h_x``,
// ``c_y = σ t h_x / h_y``, zero on inactive copper), the two unit Q1 sheet
// matrices, the active and free node masks, the via adjacency, the Jacobi
// diagonal and, for the two-level preconditioner, the coarse space; both the
// FP64 operator of the outer refinement and the FP32 operator of the inner
// PCG are views of these arrays.  The right-hand side (current shares plus
// the lifting of voltage terminals), the solve and the post-processing
// (element fields, Joule losses, via currents, terminal currents) run here
// too, so a facade only converts terminals to flat node indices.
//
// Nothing depends on the team size: every node and element is written by
// one thread from its own inputs, and the sums are lane sums.
#pragma once

#include <cstdint>
#include <vector>

#include "pcbcore/fem/layered_dc.hpp"
#include "pcbcore/fem/mpir.hpp"
#include "pcbcore/fem/two_level.hpp"

namespace pcbcore::fem {

// Non-owning description of one conductor role's copper.
struct LayeredDCMeshView {
    int layers{0};
    int rows{0};  // elements; nodes are (layers, rows + 1, cols + 1)
    int cols{0};
    const std::uint8_t* element_active{nullptr};  // (layers, rows, cols)
    const double* layer_thickness_m{nullptr};     // (layers,)
    const double* pitch_x_m{nullptr};             // (cols,)
    const double* pitch_y_m{nullptr};             // (rows,)
    const double* conductivity_s_per_m{nullptr};  // (layers, rows, cols)
};

// Flat node indices of the vias' two ends and their conductances, in via order.
struct ViaLinksView {
    std::int64_t count{0};
    const std::int64_t* lower{nullptr};
    const std::int64_t* upper{nullptr};
    const double* conductance_s{nullptr};
};

// Node groups in CSR form: group ``k`` is ``nodes[offsets[k] .. offsets[k+1])``.
struct NodeGroupsView {
    std::int64_t count{0};
    const std::int64_t* offsets{nullptr};
    const std::int64_t* nodes{nullptr};
};

struct LayeredDCPost {
    std::vector<double> electric_field;    // (layers, rows, cols, 2), V/m
    std::vector<double> current_density;   // (layers, rows, cols, 2), A/m^2
    double max_current_density{0.0};       // over active elements
    std::vector<double> element_joule_loss;  // (layers, rows, cols), W
    std::vector<double> via_current;       // (vias,), A, lower to upper
    std::vector<double> via_joule_loss;    // (vias,), W
    double joule_loss{0.0};
};

class LayeredDCSystem {
public:
    // ``reference_node`` < 0 when there is none; ``dirichlet_nodes`` are
    // held at the potentials the right-hand side lifts.  ``two_level`` adds
    // the patch-constant coarse correction; ``block`` <= 0 picks the patch
    // with choose_block_size.  Throws InvalidInput for a fixed node off the
    // copper or a free node without conductance.
    LayeredDCSystem(const LayeredDCMeshView& mesh, const ViaLinksView& vias, std::int64_t reference_node,
                    const std::int64_t* dirichlet_nodes, std::int64_t dirichlet_count, bool two_level, int block,
                    int threads);

    [[nodiscard]] int layers() const noexcept { return layers_; }
    [[nodiscard]] int rows() const noexcept { return rows_; }
    [[nodiscard]] int cols() const noexcept { return cols_; }
    [[nodiscard]] std::int64_t size() const noexcept { return size_; }
    [[nodiscard]] std::int64_t via_count() const noexcept { return static_cast<std::int64_t>(via_g_.size()); }
    [[nodiscard]] bool two_level() const noexcept { return two_level_; }
    [[nodiscard]] int block() const noexcept { return block_; }

    [[nodiscard]] const std::vector<std::uint8_t>& active_nodes() const noexcept { return active_; }
    [[nodiscard]] const std::vector<std::uint8_t>& free_nodes() const noexcept { return free_; }
    [[nodiscard]] const std::vector<double>& coefficients() const noexcept { return coef_; }
    [[nodiscard]] const std::vector<double>& unit() const noexcept { return unit_; }
    [[nodiscard]] const std::vector<double>& diagonal() const noexcept { return diagonal_; }
    [[nodiscard]] const std::vector<double>& conductivity() const noexcept { return conductivity_; }
    [[nodiscard]] const CoarseSpace& coarse() const noexcept { return coarse_; }
    // Bytes of the arrays the FP32 inner solve reads (operator, masks, vias,
    // diagonal, coarse inverse).
    [[nodiscard]] std::int64_t low_bytes() const noexcept;

    // A x with fixed rows as the identity (what the MPIR solves).
    void apply_high(const double* x, double* y, int threads) const;
    void apply_low(const float* x, float* y, int threads) const;
    // K v: the current each node sends into copper and vias, fixed rows too.
    void apply_full(const double* x, double* y, int threads) const;

    // Each group's nodes held at its potential, zero elsewhere; a later group
    // overwrites an earlier one on a shared node.  Throws InvalidInput when a
    // node is free or off the copper (index of the group in the message).
    [[nodiscard]] std::vector<double> dirichlet_potential(const NodeGroupsView& voltage, const double* voltage_v) const;

    // Current shares on free rows, the potential on fixed rows, and the
    // lifting ``f - K g`` of the voltage groups.
    [[nodiscard]] std::vector<double> build_rhs(const NodeGroupsView& current, const double* current_a,
                                                const NodeGroupsView& voltage, const double* voltage_v,
                                                int threads) const;

    // The current each group drives into the copper (NaN potentials as zero).
    [[nodiscard]] std::vector<double> group_currents(const double* potential, const NodeGroupsView& groups,
                                                     int threads) const;

    // ``x`` holds the initial guess and receives the solution.
    [[nodiscard]] MpirResult solve(const double* rhs, double* x, const MpirConfig& config, int threads) const;

    [[nodiscard]] std::vector<double> element_electric_field(const double* potential, int threads) const;
    [[nodiscard]] std::vector<double> element_joule_loss(const double* potential, int threads) const;
    [[nodiscard]] std::vector<double> via_current(const double* potential) const;
    [[nodiscard]] std::vector<double> via_joule_loss(const double* potential) const;
    [[nodiscard]] LayeredDCPost post_process(const double* potential, int threads) const;

private:
    [[nodiscard]] layered_dc::OperatorView<double> high_view() const noexcept;
    [[nodiscard]] layered_dc::OperatorView<double> full_view() const noexcept;
    [[nodiscard]] layered_dc::OperatorView<float> low_view() const noexcept;
    void check_nodes(const NodeGroupsView& groups, const char* what) const;

    int layers_{0};
    int rows_{0};
    int cols_{0};
    std::int64_t size_{0};
    bool two_level_{true};
    int block_{1};

    std::vector<double> pitch_x_;
    std::vector<double> pitch_y_;
    std::vector<double> conductivity_;
    std::vector<std::uint8_t> element_active_;

    std::vector<double> coef_;  // (2, layers, rows, cols)
    std::vector<double> unit_;  // (2, 4, 4)
    std::vector<std::uint8_t> active_;
    std::vector<std::uint8_t> free_;
    std::vector<std::uint8_t> all_free_;
    std::vector<double> free_mask_;
    std::vector<double> ones_;

    std::vector<std::int64_t> via_lower_;
    std::vector<std::int64_t> via_upper_;
    std::vector<double> via_g_;
    std::vector<std::int64_t> via_ptr_;
    std::vector<std::int64_t> via_nbr_;
    std::vector<double> via_link_g_;

    std::vector<double> diagonal_;
    CoarseSpace coarse_;

    std::vector<float> coef_low_;
    std::vector<float> unit_low_;
    std::vector<float> free_mask_low_;
    std::vector<float> via_link_g_low_;
    std::vector<float> diagonal_low_;
    std::vector<float> coarse_inverse_low_;
};

}  // namespace pcbcore::fem
