#include "pcbcore/thermal/thermal_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "pcbcore/errors.hpp"
#include "pcbcore/fem/two_level.hpp"
#include "pcbcore/lane_sum.hpp"

namespace pcbcore::thermal {

namespace {

constexpr double kStiffness1D[2][2] = {{1.0, -1.0}, {-1.0, 1.0}};
constexpr double kMass1D[2][2] = {{2.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 6.0}};

// Local corners (4 dz + 2 dy + dx) of each face direction, FACE_DIRECTIONS order.
constexpr int kFaceCorners[6][4] = {
    {0, 2, 4, 6},  // -x
    {1, 3, 5, 7},  // +x
    {0, 1, 4, 5},  // -y
    {2, 3, 6, 7},  // +y
    {0, 1, 2, 3},  // -z
    {4, 5, 6, 7},  // +z
};

template <typename To>
[[nodiscard]] std::vector<To> cast_all(const std::vector<double>& values) {
    std::vector<To> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<To>(values[i]);
    }
    return out;
}

// Faces of one direction of a boundary: the slab of a Top/Bottom boundary
// on active elements, the exposed faces of an Exposed one.
[[nodiscard]] std::vector<std::uint8_t> boundary_faces(const ThermalMesh& mesh, const FaceBoundary& boundary,
                                                       const int direction) {
    if (boundary.kind == FaceKind::Exposed) {
        return mesh.exposed(direction);
    }
    std::vector<std::uint8_t> faces(static_cast<std::size_t>(mesh.elements()), 0U);
    const int slab = boundary.kind == FaceKind::Top ? mesh.slabs - 1 : 0;
    for (int r = 0; r < mesh.rows; ++r) {
        for (int c = 0; c < mesh.cols; ++c) {
            const Index e = mesh.element(slab, r, c);
            faces[e] = mesh.active[e];
        }
    }
    return faces;
}

// Spread one weight per element face equally onto its four nodes, corner by
// corner as _lump_faces_onto_nodes does, and add the result to ``target``.
void add_lumped(const ThermalMesh& mesh, const std::vector<double>& face, const int direction,
                std::vector<double>& target) {
    std::vector<double> lumped(static_cast<std::size_t>(mesh.nodes()), 0.0);
    for (const int corner : kFaceCorners[direction]) {
        for (int s = 0; s < mesh.slabs; ++s) {
            for (int r = 0; r < mesh.rows; ++r) {
                for (int c = 0; c < mesh.cols; ++c) {
                    lumped[mesh.corner_node(s, r, c, corner)] += face[mesh.element(s, r, c)] / 4.0;
                }
            }
        }
    }
    for (std::size_t n = 0; n < lumped.size(); ++n) {
        target[n] += lumped[n];
    }
}

void newton_step(const double emissivity, double surface, const double ambient, double& coefficient,
                 double& effective) {
    if (!(std::isfinite(surface) && surface > 0.0)) {
        surface = ambient;
    }
    const double cubic = std::pow(surface, 3.0);
    coefficient = 4.0 * emissivity * kStefanBoltzmann * cubic;
    effective = surface - (std::pow(surface, 4.0) - std::pow(ambient, 4.0)) / (4.0 * cubic);
}

[[nodiscard]] int team_for(const int threads, const Index work) noexcept {
    return static_cast<int>(std::max<Index>(1, std::min<Index>(threads, std::max<Index>(1, work))));
}

}  // namespace

bool ThermalMesh::full() const noexcept {
    return std::all_of(active.begin(), active.end(), [](const std::uint8_t a) { return a != 0U; });
}

std::vector<std::uint8_t> ThermalMesh::active_nodes() const {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(nodes()), 0U);
    for (int s = 0; s < slabs; ++s) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (active[element(s, r, c)] == 0U) {
                    continue;
                }
                for (int corner = 0; corner < 8; ++corner) {
                    out[corner_node(s, r, c, corner)] = 1U;
                }
            }
        }
    }
    return out;
}

std::vector<std::uint8_t> ThermalMesh::exposed(const int direction) const {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(elements()), 0U);
    const int step = (direction % 2 == 1) ? 1 : -1;
    const int axis = direction / 2;  // 0 x, 1 y, 2 z
    for (int s = 0; s < slabs; ++s) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const Index e = element(s, r, c);
                if (active[e] == 0U) {
                    continue;
                }
                int ns = s;
                int nr = r;
                int nc = c;
                (axis == 0 ? nc : axis == 1 ? nr : ns) += step;
                const bool inside = ns >= 0 && ns < slabs && nr >= 0 && nr < rows && nc >= 0 && nc < cols;
                out[e] = (inside && active[element(ns, nr, nc)] != 0U) ? 0U : 1U;
            }
        }
    }
    return out;
}

double ThermalMesh::face_area(const int direction, const int s, const int r, const int c) const noexcept {
    switch (direction / 2) {
        case 2:
            return pitch_x[c] * pitch_y[r];
        case 1:
            return pitch_x[c] * thickness[s];
        default:
            return pitch_y[r] * thickness[s];
    }
}

LumpedRobin lump(const ThermalMesh& mesh, const FaceBoundary& boundary) {
    LumpedRobin out;
    out.weights.assign(static_cast<std::size_t>(mesh.nodes()), 0.0);
    out.rhs.assign(static_cast<std::size_t>(mesh.nodes()), 0.0);
    out.mean_ambient = boundary.mean_ambient;
    std::vector<double> face(static_cast<std::size_t>(mesh.elements()), 0.0);
    std::vector<double> load(static_cast<std::size_t>(mesh.elements()), 0.0);
    for (const int direction : boundary.directions) {
        if (direction < 0 || direction > 5) {
            throw InvalidInput("face directions are 0..5 (-x, +x, -y, +y, -z, +z)");
        }
        const std::vector<std::uint8_t> faces = boundary_faces(mesh, boundary, direction);
        for (int s = 0; s < mesh.slabs; ++s) {
            for (int r = 0; r < mesh.rows; ++r) {
                for (int c = 0; c < mesh.cols; ++c) {
                    const Index e = mesh.element(s, r, c);
                    face[e] = (boundary.coefficient[e] * mesh.face_area(direction, s, r, c)) *
                              (faces[e] != 0U ? 1.0 : 0.0);
                    load[e] = face[e] * boundary.ambient[e];
                }
            }
        }
        add_lumped(mesh, face, direction, out.weights);
        add_lumped(mesh, load, direction, out.rhs);
    }
    return out;
}

FaceBoundary linearize(const ThermalMesh& mesh, const FaceBoundary& radiation, const double* const temperature) {
    FaceBoundary out;
    out.kind = radiation.kind;
    out.directions = radiation.directions;
    out.coefficient.assign(static_cast<std::size_t>(mesh.elements()), 0.0);
    out.ambient.assign(static_cast<std::size_t>(mesh.elements()), 0.0);
    const int nr = mesh.rows + 1;
    const int nc = mesh.cols + 1;
    if (radiation.kind == FaceKind::Exposed) {
        for (int s = 0; s < mesh.slabs; ++s) {
            for (int r = 0; r < mesh.rows; ++r) {
                for (int c = 0; c < mesh.cols; ++c) {
                    const Index e = mesh.element(s, r, c);
                    double total = 0.0;
                    for (int corner = 0; corner < 8; ++corner) {
                        total += temperature[mesh.corner_node(s, r, c, corner)];
                    }
                    newton_step(radiation.coefficient[e], total / 8.0, radiation.ambient[e], out.coefficient[e],
                                out.ambient[e]);
                }
            }
        }
        out.mean_ambient = lane_sum(mesh.elements(), [&](const std::ptrdiff_t e) { return out.ambient[e]; }) /
                           static_cast<double>(mesh.elements());
        return out;
    }
    const int slab = radiation.kind == FaceKind::Top ? mesh.slabs - 1 : 0;
    const Index layer = static_cast<Index>(radiation.kind == FaceKind::Top ? mesh.slabs : 0) * nr * nc;
    for (int r = 0; r < mesh.rows; ++r) {
        for (int c = 0; c < mesh.cols; ++c) {
            const Index at = layer + static_cast<Index>(r) * nc + c;
            const double surface =
                0.25 * (((temperature[at] + temperature[at + 1]) + temperature[at + nc]) + temperature[at + nc + 1]);
            const Index e = mesh.element(slab, r, c);
            newton_step(radiation.coefficient[e], surface, radiation.ambient[e], out.coefficient[e], out.ambient[e]);
        }
    }
    const Index base = mesh.element(slab, 0, 0);
    const Index faces = static_cast<Index>(mesh.rows) * mesh.cols;
    out.mean_ambient = lane_sum(faces, [&](const std::ptrdiff_t i) { return out.ambient[base + i]; }) /
                       static_cast<double>(faces);
    return out;
}

std::vector<double> nodal_load(const ThermalMesh& mesh, const double* const element_heat,
                               const double* const nodal_heat, const Index sources, const Index* const source_offsets,
                               const Index* const source_nodes, const double* const source_power) {
    std::vector<double> load(nodal_heat, nodal_heat + mesh.nodes());
    for (int corner = 0; corner < 8; ++corner) {
        for (int s = 0; s < mesh.slabs; ++s) {
            for (int r = 0; r < mesh.rows; ++r) {
                for (int c = 0; c < mesh.cols; ++c) {
                    load[mesh.corner_node(s, r, c, corner)] += element_heat[mesh.element(s, r, c)] / 8.0;
                }
            }
        }
    }
    for (Index k = 0; k < sources; ++k) {
        const Index count = source_offsets[k + 1] - source_offsets[k];
        if (count < 1) {
            throw InvalidInput("a heat source needs at least one node");
        }
        const double per_node = source_power[k] / static_cast<double>(count);
        for (Index i = source_offsets[k]; i < source_offsets[k + 1]; ++i) {
            if (source_nodes[i] < 0 || source_nodes[i] >= mesh.nodes()) {
                throw InvalidInput("a heat source node lies outside the mesh");
            }
            load[source_nodes[i]] += per_node;
        }
    }
    return load;
}

ThermalSystem::ThermalSystem(const ThermalProblem& problem, std::vector<LumpedRobin> robin,
                             const double* const capacity_per_s, const bool two_level, const int block,
                             const int threads)
    : problem_(&problem), two_level_(two_level), robin_(std::move(robin)) {
    const ThermalMesh& mesh = problem.mesh;
    if (mesh.slabs < 1 || mesh.rows < 1 || mesh.cols < 1) {
        throw InvalidInput("the thermal mesh needs at least one element");
    }
    size_ = mesh.nodes();
    const Index elements = mesh.elements();

    // a_x = k_in h_y h_z / h_x, a_y = k_in h_x h_z / h_y, a_z = k_z h_x h_y / h_z.
    coef_.assign(static_cast<std::size_t>(3 * elements), 0.0);
    for (int s = 0; s < mesh.slabs; ++s) {
        const double hz = mesh.thickness[s];
        for (int r = 0; r < mesh.rows; ++r) {
            const double hy = mesh.pitch_y[r];
            for (int c = 0; c < mesh.cols; ++c) {
                const double hx = mesh.pitch_x[c];
                const Index e = mesh.element(s, r, c);
                coef_[e] = ((mesh.k_in[e] * hy) * hz) / hx;
                coef_[elements + e] = ((mesh.k_in[e] * hx) * hz) / hy;
                coef_[2 * elements + e] = ((mesh.k_z[e] * hx) * hy) / hz;
            }
        }
    }
    // U_x = kron(M, kron(M, S)), U_y = kron(M, kron(S, M)), U_z = kron(S, kron(M, M)).
    unit_.assign(192, 0.0);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            const int zi = i >> 2, yi = (i >> 1) & 1, xi = i & 1;
            const int zj = j >> 2, yj = (j >> 1) & 1, xj = j & 1;
            unit_[8 * i + j] = kMass1D[zi][zj] * (kMass1D[yi][yj] * kStiffness1D[xi][xj]);
            unit_[64 + 8 * i + j] = kMass1D[zi][zj] * (kStiffness1D[yi][yj] * kMass1D[xi][xj]);
            unit_[128 + 8 * i + j] = kStiffness1D[zi][zj] * (kMass1D[yi][yj] * kMass1D[xi][xj]);
        }
    }

    active_ = mesh.active_nodes();
    free_.assign(static_cast<std::size_t>(size_), 0U);
    fixed_temperature_.assign(static_cast<std::size_t>(size_), 0.0);
    for (Index n = 0; n < size_; ++n) {
        const bool fixed = problem.fixed_mask[n] != 0U;
        free_[n] = (!fixed && active_[n] != 0U) ? 1U : 0U;
        fixed_temperature_[n] = (fixed && active_[n] != 0U) ? problem.fixed_values[n] : 0.0;
    }
    free_mask_.assign(static_cast<std::size_t>(size_), 0.0);
    for (Index n = 0; n < size_; ++n) {
        free_mask_[n] = free_[n] != 0U ? 1.0 : 0.0;
    }
    all_free_.assign(static_cast<std::size_t>(size_), 1U);
    ones_.assign(static_cast<std::size_t>(size_), 1.0);
    zeros_.assign(static_cast<std::size_t>(size_), 0.0);

    robin_only_.assign(static_cast<std::size_t>(size_), 0.0);
    robin_rhs_.assign(static_cast<std::size_t>(size_), 0.0);
    for (const LumpedRobin& boundary : robin_) {
        for (Index n = 0; n < size_; ++n) {
            robin_only_[n] += boundary.weights[n];
            robin_rhs_[n] += boundary.rhs[n];
        }
    }
    capacity_.assign(static_cast<std::size_t>(size_), 0.0);
    if (capacity_per_s != nullptr) {
        for (Index n = 0; n < size_; ++n) {
            if (!std::isfinite(capacity_per_s[n]) || capacity_per_s[n] < 0.0) {
                throw InvalidInput("capacity_per_s must be finite and non-negative");
            }
            capacity_[n] = capacity_per_s[n];
        }
    }
    robin_total_.assign(static_cast<std::size_t>(size_), 0.0);
    for (Index n = 0; n < size_; ++n) {
        robin_total_[n] = robin_only_[n] + capacity_[n];
    }

    // Diagonal, summed as _build_diagonal: corners 0..7, then the Robin term.
    diagonal_.assign(static_cast<std::size_t>(size_), 0.0);
    for (int corner = 0; corner < 8; ++corner) {
        const int k = 9 * corner;
        for (int s = 0; s < mesh.slabs; ++s) {
            for (int r = 0; r < mesh.rows; ++r) {
                for (int c = 0; c < mesh.cols; ++c) {
                    const Index e = mesh.element(s, r, c);
                    diagonal_[mesh.corner_node(s, r, c, corner)] +=
                        (coef_[e] * unit_[k] + coef_[elements + e] * unit_[64 + k]) +
                        coef_[2 * elements + e] * unit_[128 + k];
                }
            }
        }
    }
    for (Index n = 0; n < size_; ++n) {
        const double value = diagonal_[n] + robin_total_[n];
        if (free_[n] != 0U && !(value > 0.0)) {
            throw InvalidInput("every free node must have positive conductance");
        }
        diagonal_[n] = free_[n] != 0U ? value : 1.0;
    }

    if (two_level_) {
        block_ = block > 0 ? block : fem::choose_block_size(mesh.slabs + 1, mesh.rows + 1, mesh.cols + 1);
        coarse_ = fem::thermal_hex::assemble_coarse(high_view(), block_, threads);
        coarse_inverse_low_ = cast_all<float>(coarse_.inverse);
    } else {
        block_ = 1;
    }
    coef_low_ = cast_all<float>(coef_);
    unit_low_ = cast_all<float>(unit_);
    robin_total_low_ = cast_all<float>(robin_total_);
    free_mask_low_ = cast_all<float>(free_mask_);
    diagonal_low_ = cast_all<float>(diagonal_);
}

fem::thermal_hex::OperatorView<double> ThermalSystem::high_view() const noexcept {
    const ThermalMesh& m = problem_->mesh;
    return {coef_.data(), unit_.data(), robin_total_.data(), free_.data(), free_mask_.data(), m.slabs, m.rows, m.cols};
}

fem::thermal_hex::OperatorView<double> ThermalSystem::stiffness_view() const noexcept {
    const ThermalMesh& m = problem_->mesh;
    return {coef_.data(), unit_.data(), zeros_.data(), all_free_.data(), ones_.data(), m.slabs, m.rows, m.cols};
}

fem::thermal_hex::OperatorView<float> ThermalSystem::low_view() const noexcept {
    const ThermalMesh& m = problem_->mesh;
    return {coef_low_.data(), unit_low_.data(), robin_total_low_.data(), free_.data(), free_mask_low_.data(),
            m.slabs, m.rows, m.cols};
}

void ThermalSystem::apply_high(const double* const x, double* const y, const int threads) const {
    fem::thermal_hex::apply_high(high_view(), x, y, threads);
}

void ThermalSystem::apply_low(const float* const x, float* const y, const int threads) const {
    fem::thermal_hex::apply_low(low_view(), x, y, threads);
}

void ThermalSystem::stiffness(const double* const x, double* const y, const int threads) const {
    fem::thermal_hex::apply_high(stiffness_view(), x, y, threads);
}

double ThermalSystem::default_reference() const {
    if (!robin_.empty()) {
        return robin_.front().mean_ambient;
    }
    const auto& fixed = problem_->fixed_mask;
    std::vector<double> values;
    for (Index n = 0; n < size_; ++n) {
        if (fixed[n] != 0U && active_[n] != 0U) {
            values.push_back(problem_->fixed_values[n]);
        }
    }
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return lane_sum(static_cast<std::ptrdiff_t>(values.size()), [&](const std::ptrdiff_t i) { return values[i]; }) /
           static_cast<double>(values.size());
}

std::vector<double> ThermalSystem::build_rhs(const double reference, const double* const previous,
                                             const int threads) const {
    std::vector<double> shifted(static_cast<std::size_t>(size_), 0.0);
    for (Index n = 0; n < size_; ++n) {
        if (free_[n] == 0U && active_[n] != 0U) {
            shifted[n] = fixed_temperature_[n] - reference;
        }
    }
    std::vector<double> boundary(static_cast<std::size_t>(size_), 0.0);
    stiffness(shifted.data(), boundary.data(), threads);
    if (previous == nullptr) {
        for (Index n = 0; n < size_; ++n) {
            if (capacity_[n] > 0.0) {
                throw InvalidInput("a transient operator needs previous_temperature_k");
            }
        }
    }
    const auto& load = problem_->load;
    std::vector<double> rhs(static_cast<std::size_t>(size_), 0.0);
    for (Index n = 0; n < size_; ++n) {
        const double robin = robin_rhs_[n] - robin_total_[n] * reference;
        const double stored = previous == nullptr ? 0.0 : capacity_[n] * (std::isfinite(previous[n]) ? previous[n] : 0.0);
        rhs[n] = free_[n] != 0U ? ((load[n] + robin) + stored) - boundary[n] : shifted[n];
    }
    return rhs;
}

fem::MpirResult ThermalSystem::solve(const double* const rhs, double* const x, const fem::MpirConfig& config,
                                     const int threads) const {
    return fem::thermal_hex::solve_mpir(low_view(), high_view(), rhs, x, diagonal_low_.data(), block_,
                                        two_level_ ? coarse_inverse_low_.data() : nullptr, config, threads);
}

std::vector<double> ThermalSystem::stored_heat(const double* const temperature, const double* const previous) const {
    std::vector<double> out(static_cast<std::size_t>(size_), 0.0);
    if (previous == nullptr) {
        return out;
    }
    for (Index n = 0; n < size_; ++n) {
        if (std::isfinite(temperature[n]) && std::isfinite(previous[n])) {
            out[n] = capacity_[n] * (temperature[n] - previous[n]);
        }
    }
    return out;
}

std::vector<double> ThermalSystem::unconstrained_residual(const double* const temperature,
                                                          const double* const previous, const int threads) const {
    std::vector<double> out(static_cast<std::size_t>(size_), 0.0);
    stiffness(temperature, out.data(), threads);
    const std::vector<double> stored = stored_heat(temperature, previous);
    const auto& load = problem_->load;
    for (Index n = 0; n < size_; ++n) {
        out[n] = (((out[n] + robin_only_[n] * temperature[n]) - robin_rhs_[n]) + stored[n]) - load[n];
    }
    return out;
}

std::vector<double> ThermalSystem::convective_heat(const double* const temperature) const {
    std::vector<double> out;
    out.reserve(robin_.size());
    for (const LumpedRobin& boundary : robin_) {
        const double removed =
            lane_sum(size_, [&](const std::ptrdiff_t n) { return boundary.weights[n] * temperature[n]; });
        const double ambient = lane_sum(size_, [&](const std::ptrdiff_t n) { return boundary.rhs[n]; });
        out.push_back(removed - ambient);
    }
    return out;
}

std::vector<double> ThermalSystem::element_heat_flux(const double* const t, const int threads) const {
    const ThermalMesh& mesh = problem_->mesh;
    std::vector<double> flux(static_cast<std::size_t>(3 * mesh.elements()), 0.0);
    const int lines = mesh.slabs * mesh.rows;
    const int team = team_for(threads, lines);
#pragma omp parallel for num_threads(team) schedule(static) if (team > 1)
    for (int line = 0; line < lines; ++line) {
        const int s = line / mesh.rows;
        const int r = line - s * mesh.rows;
        for (int c = 0; c < mesh.cols; ++c) {
            double v[8];
            for (int corner = 0; corner < 8; ++corner) {
                v[corner] = t[mesh.corner_node(s, r, c, corner)];
            }
            const double x_plus = ((v[1] + v[3]) + v[5]) + v[7];
            const double x_minus = ((v[0] + v[2]) + v[4]) + v[6];
            const double y_plus = ((v[2] + v[3]) + v[6]) + v[7];
            const double y_minus = ((v[0] + v[1]) + v[4]) + v[5];
            const double z_plus = ((v[4] + v[5]) + v[6]) + v[7];
            const double z_minus = ((v[0] + v[1]) + v[2]) + v[3];
            const Index e = mesh.element(s, r, c);
            const double gx = (x_plus - x_minus) / (4.0 * mesh.pitch_x[c]);
            const double gy = (y_plus - y_minus) / (4.0 * mesh.pitch_y[r]);
            const double gz = (z_plus - z_minus) / (4.0 * mesh.thickness[s]);
            flux[3 * e] = -mesh.k_in[e] * gx;
            flux[3 * e + 1] = -mesh.k_in[e] * gy;
            flux[3 * e + 2] = -mesh.k_z[e] * gz;
        }
    }
    return flux;
}

}  // namespace pcbcore::thermal
