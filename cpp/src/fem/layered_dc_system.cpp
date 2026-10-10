#include "pcbcore/fem/layered_dc_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "pcbcore/errors.hpp"
#include "pcbcore/lane_sum.hpp"

namespace pcbcore::fem {

namespace {

using Index = std::int64_t;

constexpr double kStiffness1D[2][2] = {{1.0, -1.0}, {-1.0, 1.0}};
constexpr double kMass1D[2][2] = {{2.0 / 6.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 6.0}};

[[nodiscard]] int team_for(const int threads, const Index work) noexcept {
    const Index cap = std::max<Index>(1, work);
    return static_cast<int>(std::max<Index>(1, std::min<Index>(threads, cap)));
}

template <typename To>
[[nodiscard]] std::vector<To> cast_all(const std::vector<double>& values) {
    std::vector<To> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<To>(values[i]);
    }
    return out;
}

}  // namespace

LayeredDCSystem::LayeredDCSystem(const LayeredDCMeshView& mesh, const ViaLinksView& vias,
                                 const std::int64_t reference_node, const std::int64_t* const dirichlet_nodes,
                                 const std::int64_t dirichlet_count, const bool two_level, const int block,
                                 const int threads)
    : layers_(mesh.layers), rows_(mesh.rows), cols_(mesh.cols), two_level_(two_level) {
    if (layers_ < 1 || rows_ < 1 || cols_ < 1) {
        throw InvalidInput("the mesh needs at least one element per layer");
    }
    if (mesh.element_active == nullptr || mesh.layer_thickness_m == nullptr || mesh.pitch_x_m == nullptr ||
        mesh.pitch_y_m == nullptr || mesh.conductivity_s_per_m == nullptr) {
        throw InvalidInput("mesh arrays are missing");
    }
    const int nr = rows_ + 1;
    const int nc = cols_ + 1;
    size_ = static_cast<Index>(layers_) * nr * nc;
    const Index elements = static_cast<Index>(layers_) * rows_ * cols_;

    pitch_x_.assign(mesh.pitch_x_m, mesh.pitch_x_m + cols_);
    pitch_y_.assign(mesh.pitch_y_m, mesh.pitch_y_m + rows_);
    conductivity_.assign(mesh.conductivity_s_per_m, mesh.conductivity_s_per_m + elements);
    element_active_.assign(mesh.element_active, mesh.element_active + elements);

    // c_x = (σ t a) h_y / h_x and c_y = (σ t a) h_x / h_y, in NumPy's order.
    coef_.assign(static_cast<std::size_t>(2 * elements), 0.0);
    for (int l = 0; l < layers_; ++l) {
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                const Index e = (static_cast<Index>(l) * rows_ + r) * cols_ + c;
                const double sheet = (conductivity_[e] * mesh.layer_thickness_m[l]) *
                                     (element_active_[e] != 0U ? 1.0 : 0.0);
                coef_[e] = (sheet * pitch_y_[r]) / pitch_x_[c];
                coef_[elements + e] = (sheet * pitch_x_[c]) / pitch_y_[r];
            }
        }
    }
    // U_x = kron(M, S), U_y = kron(S, M); local node index 2 dy + dx.
    unit_.assign(32, 0.0);
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            for (int ey = 0; ey < 2; ++ey) {
                for (int ex = 0; ex < 2; ++ex) {
                    const int at = 4 * (2 * dy + dx) + (2 * ey + ex);
                    unit_[at] = kMass1D[dy][ey] * kStiffness1D[dx][ex];
                    unit_[16 + at] = kStiffness1D[dy][ey] * kMass1D[dx][ex];
                }
            }
        }
    }

    active_.assign(static_cast<std::size_t>(size_), 0U);
    for (int l = 0; l < layers_; ++l) {
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                if (element_active_[(static_cast<Index>(l) * rows_ + r) * cols_ + c] == 0U) {
                    continue;
                }
                const Index corner = (static_cast<Index>(l) * nr + r) * nc + c;
                active_[corner] = active_[corner + 1] = 1U;
                active_[corner + nc] = active_[corner + nc + 1] = 1U;
            }
        }
    }

    via_lower_.assign(vias.lower, vias.lower + vias.count);
    via_upper_.assign(vias.upper, vias.upper + vias.count);
    via_g_.assign(vias.conductance_s, vias.conductance_s + vias.count);
    for (Index k = 0; k < vias.count; ++k) {
        if (via_lower_[k] < 0 || via_lower_[k] >= size_ || via_upper_[k] < 0 || via_upper_[k] >= size_) {
            throw InvalidInput("a via ends outside the mesh");
        }
        active_[via_lower_[k]] = 1U;
        active_[via_upper_[k]] = 1U;
    }
    // Node-owned CSR: each via under both ends; under one node the vias that
    // start there come first, then those that end there, each in via order.
    via_ptr_.assign(static_cast<std::size_t>(size_ + 1), 0);
    for (Index k = 0; k < vias.count; ++k) {
        ++via_ptr_[via_lower_[k] + 1];
        ++via_ptr_[via_upper_[k] + 1];
    }
    for (Index n = 0; n < size_; ++n) {
        via_ptr_[n + 1] += via_ptr_[n];
    }
    via_nbr_.assign(static_cast<std::size_t>(2 * vias.count), 0);
    via_link_g_.assign(static_cast<std::size_t>(2 * vias.count), 0.0);
    {
        std::vector<Index> fill(via_ptr_.begin(), via_ptr_.end() - 1);
        for (Index k = 0; k < vias.count; ++k) {
            const Index at = fill[via_lower_[k]]++;
            via_nbr_[at] = via_upper_[k];
            via_link_g_[at] = via_g_[k];
        }
        for (Index k = 0; k < vias.count; ++k) {
            const Index at = fill[via_upper_[k]]++;
            via_nbr_[at] = via_lower_[k];
            via_link_g_[at] = via_g_[k];
        }
    }

    if (reference_node < 0 && dirichlet_count == 0) {
        throw InvalidInput("pass a reference_node or dirichlet_nodes to fix the potential");
    }
    free_.assign(static_cast<std::size_t>(size_), 0U);
    for (Index n = 0; n < size_; ++n) {
        free_[n] = active_[n];
    }
    if (reference_node >= 0) {
        if (reference_node >= size_ || active_[reference_node] == 0U) {
            throw InvalidInput("reference_node must lie on active copper or a via endpoint");
        }
        free_[reference_node] = 0U;
    }
    for (Index k = 0; k < dirichlet_count; ++k) {
        const Index node = dirichlet_nodes[k];
        if (node < 0 || node >= size_ || active_[node] == 0U) {
            throw InvalidInput("dirichlet_nodes must lie on active copper or via endpoints");
        }
        free_[node] = 0U;
    }
    free_mask_.assign(static_cast<std::size_t>(size_), 0.0);
    for (Index n = 0; n < size_; ++n) {
        free_mask_[n] = free_[n] != 0U ? 1.0 : 0.0;
    }
    all_free_.assign(static_cast<std::size_t>(size_), 1U);
    ones_.assign(static_cast<std::size_t>(size_), 1.0);

    // Jacobi diagonal, summed as pcb.py's _build_diagonal does: the corner
    // terms of the elements at (r, c), (r, c-1), (r-1, c), (r-1, c-1), then
    // the vias in CSR order.
    diagonal_.assign(static_cast<std::size_t>(size_), 1.0);
    const double* const ux = unit_.data();
    const double* const uy = unit_.data() + 16;
    for (int l = 0; l < layers_; ++l) {
        for (int y = 0; y < nr; ++y) {
            for (int x = 0; x < nc; ++x) {
                const Index node = (static_cast<Index>(l) * nr + y) * nc + x;
                double d = 0.0;
                const auto corner = [&](const int ey, const int ex, const int local) {
                    if (ey < 0 || ey >= rows_ || ex < 0 || ex >= cols_) {
                        return;
                    }
                    const Index e = (static_cast<Index>(l) * rows_ + ey) * cols_ + ex;
                    d += coef_[e] * ux[5 * local] + coef_[elements + e] * uy[5 * local];
                };
                corner(y, x, 0);
                corner(y, x - 1, 1);
                corner(y - 1, x, 2);
                corner(y - 1, x - 1, 3);
                for (Index k = via_ptr_[node]; k < via_ptr_[node + 1]; ++k) {
                    d += via_link_g_[k];
                }
                if (free_[node] != 0U) {
                    if (!(d > 0.0)) {
                        throw InvalidInput("every free node must have positive conductivity coupling");
                    }
                    diagonal_[node] = d;
                }
            }
        }
    }

    if (two_level_) {
        block_ = block > 0 ? block : choose_block_size(layers_, nr, nc);
        coarse_ = layered_dc::assemble_coarse(high_view(), block_, threads);
        coarse_inverse_low_ = cast_all<float>(coarse_.inverse);
    } else {
        block_ = 1;
    }
    coef_low_ = cast_all<float>(coef_);
    unit_low_ = cast_all<float>(unit_);
    free_mask_low_ = cast_all<float>(free_mask_);
    via_link_g_low_ = cast_all<float>(via_link_g_);
    diagonal_low_ = cast_all<float>(diagonal_);
}

layered_dc::OperatorView<double> LayeredDCSystem::high_view() const noexcept {
    return {coef_.data(), unit_.data(), free_.data(), free_mask_.data(), via_ptr_.data(),
            via_nbr_.data(), via_link_g_.data(), layers_, rows_, cols_};
}

layered_dc::OperatorView<double> LayeredDCSystem::full_view() const noexcept {
    return {coef_.data(), unit_.data(), all_free_.data(), ones_.data(), via_ptr_.data(),
            via_nbr_.data(), via_link_g_.data(), layers_, rows_, cols_};
}

layered_dc::OperatorView<float> LayeredDCSystem::low_view() const noexcept {
    return {coef_low_.data(), unit_low_.data(), free_.data(), free_mask_low_.data(), via_ptr_.data(),
            via_nbr_.data(), via_link_g_low_.data(), layers_, rows_, cols_};
}

std::int64_t LayeredDCSystem::low_bytes() const noexcept {
    const auto bytes = [](const auto& values) {
        return static_cast<std::int64_t>(values.size() * sizeof(values[0]));
    };
    return bytes(coef_low_) + bytes(unit_low_) + bytes(free_) + bytes(free_mask_low_) + bytes(via_ptr_) +
           bytes(via_nbr_) + bytes(via_link_g_low_) + bytes(diagonal_low_) + bytes(coarse_inverse_low_);
}

void LayeredDCSystem::apply_high(const double* const x, double* const y, const int threads) const {
    layered_dc::apply_high(high_view(), x, y, threads);
}

void LayeredDCSystem::apply_low(const float* const x, float* const y, const int threads) const {
    layered_dc::apply_low(low_view(), x, y, threads);
}

void LayeredDCSystem::apply_full(const double* const x, double* const y, const int threads) const {
    layered_dc::apply_high(full_view(), x, y, threads);
}

void LayeredDCSystem::check_nodes(const NodeGroupsView& groups, const char* const what) const {
    if (groups.count == 0) {
        return;
    }
    if (groups.offsets == nullptr || groups.offsets[0] != 0) {
        throw InvalidInput(std::string(what) + " offsets must start at zero");
    }
    for (Index k = 0; k < groups.count; ++k) {
        if (groups.offsets[k + 1] <= groups.offsets[k]) {
            throw InvalidInput(std::string(what) + " " + std::to_string(k) + " has no nodes");
        }
        for (Index i = groups.offsets[k]; i < groups.offsets[k + 1]; ++i) {
            const Index node = groups.nodes[i];
            if (node < 0 || node >= size_ || active_[node] == 0U) {
                throw InvalidInput(std::string(what) + " " + std::to_string(k) + " contains an inactive node");
            }
        }
    }
}

std::vector<double> LayeredDCSystem::dirichlet_potential(const NodeGroupsView& voltage,
                                                         const double* const voltage_v) const {
    check_nodes(voltage, "voltage terminal");
    std::vector<double> potential(static_cast<std::size_t>(size_), 0.0);
    for (Index k = 0; k < voltage.count; ++k) {
        for (Index i = voltage.offsets[k]; i < voltage.offsets[k + 1]; ++i) {
            const Index node = voltage.nodes[i];
            if (free_[node] != 0U) {
                throw InvalidInput("voltage terminal " + std::to_string(k) +
                                   " has a node that is not a Dirichlet node of this operator");
            }
            potential[node] = voltage_v[k];
        }
    }
    return potential;
}

std::vector<double> LayeredDCSystem::build_rhs(const NodeGroupsView& current, const double* const current_a,
                                               const NodeGroupsView& voltage, const double* const voltage_v,
                                               const int threads) const {
    check_nodes(current, "current terminal");
    std::vector<double> rhs(static_cast<std::size_t>(size_), 0.0);
    for (Index k = 0; k < current.count; ++k) {
        const double share = current_a[k] / static_cast<double>(current.offsets[k + 1] - current.offsets[k]);
        for (Index i = current.offsets[k]; i < current.offsets[k + 1]; ++i) {
            rhs[current.nodes[i]] += share;
        }
    }
    // The reference equation is redundant in the balanced Neumann system;
    // the gauge V_ref = 0 replaces it.
    for (Index n = 0; n < size_; ++n) {
        if (free_[n] == 0U) {
            rhs[n] = 0.0;
        }
    }
    if (voltage.count > 0) {
        // Lifting: free rows see f - K g, fixed rows read V = g directly.
        const std::vector<double> fixed = dirichlet_potential(voltage, voltage_v);
        std::vector<double> lifted(static_cast<std::size_t>(size_), 0.0);
        apply_full(fixed.data(), lifted.data(), threads);
        for (Index n = 0; n < size_; ++n) {
            rhs[n] = free_[n] != 0U ? rhs[n] - lifted[n] : fixed[n];
        }
    }
    return rhs;
}

std::vector<double> LayeredDCSystem::group_currents(const double* const potential, const NodeGroupsView& groups,
                                                    const int threads) const {
    check_nodes(groups, "voltage terminal");
    std::vector<double> clean(potential, potential + size_);
    for (double& value : clean) {
        if (std::isnan(value)) {
            value = 0.0;
        }
    }
    std::vector<double> injected(static_cast<std::size_t>(size_), 0.0);
    apply_full(clean.data(), injected.data(), threads);
    std::vector<double> out(static_cast<std::size_t>(groups.count), 0.0);
    for (Index k = 0; k < groups.count; ++k) {
        const Index begin = groups.offsets[k];
        out[k] = lane_sum(groups.offsets[k + 1] - begin,
                          [&](const std::ptrdiff_t i) { return injected[groups.nodes[begin + i]]; });
    }
    return out;
}

MpirResult LayeredDCSystem::solve(const double* const rhs, double* const x, const MpirConfig& config,
                                  const int threads) const {
    return layered_dc::solve_mpir(low_view(), high_view(), rhs, x, diagonal_low_.data(), block_,
                                  two_level_ ? coarse_inverse_low_.data() : nullptr, config, threads);
}

std::vector<double> LayeredDCSystem::element_electric_field(const double* const potential, const int threads) const {
    const int nr = rows_ + 1;
    const int nc = cols_ + 1;
    const int lines = layers_ * rows_;
    std::vector<double> field(static_cast<std::size_t>(2) * layers_ * rows_ * cols_, 0.0);
    const int team = team_for(threads, lines);
#pragma omp parallel for num_threads(team) schedule(static) if (team > 1)
    for (int line = 0; line < lines; ++line) {
        const int l = line / rows_;
        const int r = line - l * rows_;
        for (int c = 0; c < cols_; ++c) {
            const Index e = static_cast<Index>(line) * cols_ + c;
            if (element_active_[e] == 0U) {
                continue;
            }
            const Index corner = (static_cast<Index>(l) * nr + r) * nc + c;
            const double v00 = potential[corner];
            const double v01 = potential[corner + 1];
            const double v10 = potential[corner + nc];
            const double v11 = potential[corner + nc + 1];
            field[2 * e] = -((v01 + v11) - (v00 + v10)) / (2.0 * pitch_x_[c]);
            field[2 * e + 1] = -((v10 + v11) - (v00 + v01)) / (2.0 * pitch_y_[r]);
        }
    }
    return field;
}

std::vector<double> LayeredDCSystem::element_joule_loss(const double* const potential, const int threads) const {
    const int nr = rows_ + 1;
    const int nc = cols_ + 1;
    const int lines = layers_ * rows_;
    const Index elements = static_cast<Index>(lines) * cols_;
    std::vector<double> loss(static_cast<std::size_t>(elements), 0.0);
    const double* const ux = unit_.data();
    const double* const uy = unit_.data() + 16;
    const int team = team_for(threads, lines);
#pragma omp parallel for num_threads(team) schedule(static) if (team > 1)
    for (int line = 0; line < lines; ++line) {
        const int l = line / rows_;
        const int r = line - l * rows_;
        for (int c = 0; c < cols_; ++c) {
            const Index e = static_cast<Index>(line) * cols_ + c;
            const Index corner = (static_cast<Index>(l) * nr + r) * nc + c;
            const double v[4] = {potential[corner], potential[corner + 1], potential[corner + nc],
                                 potential[corner + nc + 1]};
            // The element quadratic form sigma t v^T K_e v.
            double qx = 0.0;
            double qy = 0.0;
            for (int i = 0; i < 4; ++i) {
                double rx = 0.0;
                double ry = 0.0;
                for (int j = 0; j < 4; ++j) {
                    rx += ux[4 * i + j] * v[j];
                    ry += uy[4 * i + j] * v[j];
                }
                qx += v[i] * rx;
                qy += v[i] * ry;
            }
            loss[e] = coef_[e] * qx + coef_[elements + e] * qy;
        }
    }
    return loss;
}

std::vector<double> LayeredDCSystem::via_current(const double* const potential) const {
    std::vector<double> current(via_g_.size(), 0.0);
    for (std::size_t k = 0; k < via_g_.size(); ++k) {
        current[k] = via_g_[k] * (potential[via_lower_[k]] - potential[via_upper_[k]]);
    }
    return current;
}

std::vector<double> LayeredDCSystem::via_joule_loss(const double* const potential) const {
    std::vector<double> loss(via_g_.size(), 0.0);
    for (std::size_t k = 0; k < via_g_.size(); ++k) {
        const double drop = potential[via_lower_[k]] - potential[via_upper_[k]];
        loss[k] = via_g_[k] * drop * drop;
    }
    return loss;
}

LayeredDCPost LayeredDCSystem::post_process(const double* const potential, const int threads) const {
    LayeredDCPost post;
    post.electric_field = element_electric_field(potential, threads);
    const Index elements = static_cast<Index>(layers_) * rows_ * cols_;
    post.current_density.assign(post.electric_field.size(), 0.0);
    // np.max over the active elements: NaN wins, no element gives zero.
    double maximum = 0.0;
    bool not_a_number = false;
    for (Index e = 0; e < elements; ++e) {
        const double jx = conductivity_[e] * post.electric_field[2 * e];
        const double jy = conductivity_[e] * post.electric_field[2 * e + 1];
        post.current_density[2 * e] = jx;
        post.current_density[2 * e + 1] = jy;
        if (element_active_[e] != 0U) {
            const double magnitude = std::sqrt(jx * jx + jy * jy);
            not_a_number = not_a_number || std::isnan(magnitude);
            maximum = std::max(maximum, magnitude);
        }
    }
    post.max_current_density = not_a_number ? std::numeric_limits<double>::quiet_NaN() : maximum;
    post.element_joule_loss = element_joule_loss(potential, threads);
    post.via_current = via_current(potential);
    post.via_joule_loss = via_joule_loss(potential);
    const auto& element = post.element_joule_loss;
    const auto& via = post.via_joule_loss;
    post.joule_loss = lane_sum(static_cast<std::ptrdiff_t>(element.size()),
                               [&](const std::ptrdiff_t i) { return element[i]; }) +
                      lane_sum(static_cast<std::ptrdiff_t>(via.size()), [&](const std::ptrdiff_t i) { return via[i]; });
    return post;
}

}  // namespace pcbcore::fem
