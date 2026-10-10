// Steady and transient heat conduction on a layered hexahedral Q1 mesh: the
// problem (mesh, fixed temperatures, loads, convective and radiating faces),
// the prepared system of one linear solve, and the solves themselves
// (thermal/matrix_free_mpir_fem/{mesh,boundaries,radiation,operator,solve,
// transient}.py).
//
// Nodes are (slabs + 1, rows + 1, cols + 1), elements (slabs, rows, cols),
// local corners ``4 dz + 2 dy + dx``.  Face directions are numbered as
// FACE_DIRECTIONS: -x, +x, -y, +y, -z, +z.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "pcbcore/fem/mpir.hpp"
#include "pcbcore/fem/thermal_hex.hpp"

namespace pcbcore::thermal {

using Index = std::int64_t;

inline constexpr double kStefanBoltzmann = 5.670374419e-8;

struct ThermalMesh {
    int slabs{0};
    int rows{0};
    int cols{0};
    std::vector<std::uint8_t> active;   // (slabs, rows, cols)
    std::vector<double> thickness;      // (slabs,)
    std::vector<double> pitch_x;        // (cols,)
    std::vector<double> pitch_y;        // (rows,)
    std::vector<double> k_in;           // (slabs, rows, cols), W/m/K
    std::vector<double> k_z;            // (slabs, rows, cols)

    [[nodiscard]] Index elements() const noexcept { return static_cast<Index>(slabs) * rows * cols; }
    [[nodiscard]] Index nodes() const noexcept { return static_cast<Index>(slabs + 1) * (rows + 1) * (cols + 1); }
    [[nodiscard]] Index element(int s, int r, int c) const noexcept {
        return (static_cast<Index>(s) * rows + r) * cols + c;
    }
    // The node at local corner ``corner`` of element (s, r, c).
    [[nodiscard]] Index corner_node(int s, int r, int c, int corner) const noexcept {
        return (static_cast<Index>(s + (corner >> 2)) * (rows + 1) + r + ((corner >> 1) & 1)) * (cols + 1) + c +
               (corner & 1);
    }
    [[nodiscard]] bool full() const noexcept;
    [[nodiscard]] std::vector<std::uint8_t> active_nodes() const;
    // Active element faces bordering a void element or the grid edge.
    [[nodiscard]] std::vector<std::uint8_t> exposed(int direction) const;
    [[nodiscard]] double face_area(int direction, int s, int r, int c) const noexcept;
};

enum class FaceKind : int { Bottom = 0, Top = 1, Exposed = 2 };

// A convection boundary (``coefficient`` the film coefficient, W/m^2/K) or a
// radiation boundary (``coefficient`` the emissivity), one value per element
// for the faces it covers: the top (bottom) slab for Top (Bottom), every
// exposed face in ``directions`` for Exposed.
struct FaceBoundary {
    FaceKind kind{FaceKind::Top};
    std::vector<int> directions;
    std::vector<double> coefficient;  // (slabs, rows, cols)
    std::vector<double> ambient;      // (slabs, rows, cols), K
    double mean_ambient{0.0};
};

// Nodal conductance ``R`` and load ``R T_amb`` of one face boundary.
struct LumpedRobin {
    std::vector<double> weights;
    std::vector<double> rhs;
    double mean_ambient{0.0};
};

[[nodiscard]] LumpedRobin lump(const ThermalMesh& mesh, const FaceBoundary& boundary);

// The Newton-linearised Robin boundary of a radiation boundary at the nodal
// temperatures: h = 4 eps sigma T^3, T_eff = T - (T^4 - T_amb^4) / (4 T^3),
// with T the face mean (Top/Bottom) or the element mean (Exposed).
[[nodiscard]] FaceBoundary linearize(const ThermalMesh& mesh, const FaceBoundary& radiation,
                                     const double* temperature);

struct ThermalProblem {
    ThermalMesh mesh;
    std::vector<std::uint8_t> fixed_mask;  // nodes
    std::vector<double> fixed_values;      // nodes, zero where not fixed
    std::vector<FaceBoundary> convection;
    std::vector<FaceBoundary> radiation;
    std::vector<double> load;              // nodal heat input, W
    // The parts ``load`` was summed from (nodal_load), kept so a coupling
    // can change one part and sum again in the same order.
    std::vector<double> element_heat;      // (slabs, rows, cols)
    std::vector<double> nodal_heat;        // nodes
    std::vector<Index> source_offsets{0};
    std::vector<Index> source_nodes;
    std::vector<double> source_power;

    void sum_load();
};

// Heat per node: ``nodal_heat`` plus an eighth of each element's heat on
// its corners plus each source's power shared over its nodes.
[[nodiscard]] std::vector<double> nodal_load(const ThermalMesh& mesh, const double* element_heat,
                                             const double* nodal_heat, Index sources, const Index* source_offsets,
                                             const Index* source_nodes, const double* source_power);

// The prepared operator of one linear solve: (K + R + C/dt) with fixed rows
// as the identity, its diagonal and coarse space, both precisions.
class ThermalSystem {
public:
    ThermalSystem(const ThermalProblem& problem, std::vector<LumpedRobin> robin, const double* capacity_per_s,
                  bool two_level, int block, int threads);

    [[nodiscard]] Index size() const noexcept { return size_; }
    [[nodiscard]] int block() const noexcept { return block_; }
    [[nodiscard]] bool two_level() const noexcept { return two_level_; }
    [[nodiscard]] const ThermalProblem& problem() const noexcept { return *problem_; }
    [[nodiscard]] const std::vector<std::uint8_t>& active_nodes() const noexcept { return active_; }
    [[nodiscard]] const std::vector<std::uint8_t>& free_nodes() const noexcept { return free_; }
    [[nodiscard]] const std::vector<double>& fixed_temperature() const noexcept { return fixed_temperature_; }
    [[nodiscard]] const std::vector<double>& coefficients() const noexcept { return coef_; }
    [[nodiscard]] const std::vector<double>& unit() const noexcept { return unit_; }
    [[nodiscard]] const std::vector<double>& robin_total() const noexcept { return robin_total_; }
    [[nodiscard]] const std::vector<double>& diagonal() const noexcept { return diagonal_; }
    [[nodiscard]] const fem::CoarseSpace& coarse() const noexcept { return coarse_; }
    [[nodiscard]] std::size_t robin_count() const noexcept { return robin_.size(); }
    // The FP32 operator, diagonal and coarse inverse the inner solve reads
    // (empty inverse without the two-level preconditioner), for a device copy.
    [[nodiscard]] fem::thermal_hex::OperatorView<float> low_view() const noexcept;
    [[nodiscard]] const std::vector<float>& diagonal_low() const noexcept { return diagonal_low_; }
    [[nodiscard]] const std::vector<float>& coarse_inverse_low() const noexcept { return coarse_inverse_low_; }

    void apply_high(const double* x, double* y, int threads) const;
    void apply_low(const float* x, float* y, int threads) const;
    // K x: conduction only, no Robin term, fixed rows too.
    void stiffness(const double* x, double* y, int threads) const;

    // Ambient of the first Robin boundary, else the mean fixed temperature.
    [[nodiscard]] double default_reference() const;
    // Right-hand side for the rise above ``reference``; ``previous`` (finite)
    // is T_n of a backward-Euler step or null in steady state.
    [[nodiscard]] std::vector<double> build_rhs(double reference, const double* previous, int threads) const;
    [[nodiscard]] fem::MpirResult solve(const double* rhs, double* x, const fem::MpirConfig& config,
                                        int threads) const;

    [[nodiscard]] std::vector<double> stored_heat(const double* temperature, const double* previous) const;
    [[nodiscard]] std::vector<double> unconstrained_residual(const double* temperature, const double* previous,
                                                             int threads) const;
    [[nodiscard]] std::vector<double> convective_heat(const double* temperature) const;
    [[nodiscard]] std::vector<double> element_heat_flux(const double* temperature, int threads) const;

private:
    [[nodiscard]] fem::thermal_hex::OperatorView<double> high_view() const noexcept;
    [[nodiscard]] fem::thermal_hex::OperatorView<double> stiffness_view() const noexcept;

    const ThermalProblem* problem_{nullptr};
    Index size_{0};
    bool two_level_{true};
    int block_{1};
    std::vector<double> coef_;  // (3, slabs, rows, cols)
    std::vector<double> unit_;  // (3, 8, 8)
    std::vector<std::uint8_t> active_;
    std::vector<std::uint8_t> free_;
    std::vector<std::uint8_t> all_free_;
    std::vector<double> free_mask_;
    std::vector<double> ones_;
    std::vector<double> zeros_;
    std::vector<double> fixed_temperature_;
    std::vector<LumpedRobin> robin_;
    std::vector<double> robin_only_;
    std::vector<double> capacity_;
    std::vector<double> robin_total_;
    std::vector<double> robin_rhs_;
    std::vector<double> diagonal_;
    fem::CoarseSpace coarse_;
    std::vector<float> coef_low_;
    std::vector<float> unit_low_;
    std::vector<float> robin_total_low_;
    std::vector<float> free_mask_low_;
    std::vector<float> diagonal_low_;
    std::vector<float> coarse_inverse_low_;
};

struct ThermalSolution {
    std::vector<double> temperature;  // nodes; NaN on inactive nodes of a carved mesh
    std::vector<double> heat_flux;    // (slabs, rows, cols, 3)
    double max_temperature{0.0};
    double min_temperature{0.0};
    double total_heat_input{0.0};
    std::vector<double> convective_heat;  // per convection boundary
    std::vector<double> radiative_heat;   // per radiation boundary
    double fixed_temperature_heat{0.0};
    double heat_balance_error{0.0};
    double stored_heat{0.0};
    std::vector<double> rise;  // the last MPIR solution (rise above its reference)
    fem::MpirResult solve;
    int radiation_iterations{0};
    bool radiation_converged{true};
    double radiation_change{0.0};
};

struct SolveOptions {
    fem::MpirConfig config;
    bool two_level{true};
    int block{0};
    std::optional<double> reference;
    int radiation_max_iterations{25};
    double radiation_tolerance{1.0e-4};
    int threads{1};
};

// One linear solve (no radiation) on a prepared system, with the
// re-reference restart and the heat budget.  ``initial`` and ``previous``
// are nodal temperatures or null.
[[nodiscard]] ThermalSolution solve_linear(const ThermalSystem& system, const SolveOptions& options,
                                           const double* initial, const double* previous);

// Steady solve, or one backward-Euler step with ``capacity_per_s``
// (C / dt per node) and ``previous`` (T_n); radiation by Newton iteration.
// ``cache`` keeps the system of a problem without radiation between calls
// on the same problem object with the same capacity (the load may change).
struct SystemCache {
    const ThermalProblem* problem{nullptr};
    bool transient{false};
    std::vector<double> capacity;
    std::shared_ptr<ThermalSystem> system;
};
[[nodiscard]] ThermalSolution solve_steady(const ThermalProblem& problem, const SolveOptions& options,
                                           const double* initial, const double* capacity_per_s,
                                           const double* previous, SystemCache* cache = nullptr);

struct TransientStep {
    int index{0};
    double time{0.0};
    double step{0.0};
    double max_temperature{0.0};
    double max_change{0.0};
    double stored_heat{0.0};
    double convective_heat{0.0};
    double radiative_heat{0.0};
    double heat_balance_error{0.0};
    int inner_iterations{0};
    int radiation_iterations{0};
    bool converged{true};
};

struct TransientSolution {
    std::vector<double> times;
    std::vector<double> temperature;  // (stored, nodes)
    std::vector<TransientStep> history;
    ThermalSolution final;
    bool reached_steady{false};
};

// Backward Euler from ``initial`` (nodes, fixed nodes already set) through
// ``times`` (0 first); ``capacity`` is the lumped nodal heat capacity, J/K.
[[nodiscard]] TransientSolution solve_transient(const ThermalProblem& problem, const SolveOptions& options,
                                                const std::vector<double>& times, const double* initial,
                                                const double* capacity, bool store_all, bool until_steady,
                                                double steady_tolerance);

}  // namespace pcbcore::thermal
