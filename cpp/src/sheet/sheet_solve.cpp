#include "pcbcore/sheet/sheet_solve.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <utility>

#include "pcbcore/errors.hpp"
#include "pcbcore/linalg/csc_builder.hpp"
#include "pcbcore/linalg/gmres.hpp"
#include "pcbcore/linalg/sparse_lu.hpp"
#include "pcbcore/sheet/convolution_operator.hpp"
#include "pcbcore/sheet/pfft_operator.hpp"

namespace pcbcore::sheet {

namespace {

using Complex = std::complex<double>;
using std::size_t;
constexpr double kPi = 3.141592653589793;

// ---------------------------------------------------------------- operators

void scatter(const std::vector<std::int8_t>& family, const std::vector<std::int64_t>& position,
             const double* const currents, double* const* const inputs) {
    for (size_t b = 0; b < family.size(); ++b) {
        inputs[family[b]][position[b]] = currents[b];
    }
}

void gather(const std::vector<std::int8_t>& family, const std::vector<std::int64_t>& position,
            const double* const* const outputs, double* const flux) {
    for (size_t b = 0; b < family.size(); ++b) {
        flux[b] = outputs[family[b]][position[b]];
    }
}

void check_positions(const std::vector<std::int8_t>& family, const std::vector<std::int64_t>& position,
                     const std::int64_t* const sizes) {
    if (family.size() != position.size()) {
        throw InvalidInput("family and position must have one entry per branch");
    }
    for (size_t b = 0; b < family.size(); ++b) {
        if (family[b] < 0 || family[b] > 2 || position[b] < 0 || position[b] >= sizes[family[b]]) {
            throw InvalidInput("branch position outside its operator family");
        }
    }
}

}  // namespace

ConvolutionFlux::ConvolutionFlux(std::shared_ptr<const ConvolutionOperator> op, std::vector<std::int8_t> family,
                                 std::vector<std::int64_t> position, const std::int64_t in_plane_size,
                                 const std::int64_t vertical_size)
    : op_(std::move(op)),
      family_(std::move(family)),
      position_(std::move(position)),
      in_plane_size_(in_plane_size),
      vertical_size_(vertical_size) {
    const std::int64_t sizes[3] = {in_plane_size_, in_plane_size_, vertical_size_};
    check_positions(family_, position_, sizes);
}

void ConvolutionFlux::apply(const double* const currents, double* const flux, const int threads) const {
    std::vector<double> x(static_cast<size_t>(in_plane_size_), 0.0);
    std::vector<double> y(static_cast<size_t>(in_plane_size_), 0.0);
    std::vector<double> z(static_cast<size_t>(vertical_size_), 0.0);
    double* inputs[3] = {x.data(), y.data(), z.data()};
    scatter(family_, position_, currents, inputs);
    std::vector<double> fx(x.size());
    std::vector<double> fy(y.size());
    std::vector<double> fz(z.size());
    const bool vertical = vertical_size_ > 0;
    op_->apply(x.data(), y.data(), vertical ? z.data() : nullptr, fx.data(), fy.data(),
               vertical ? fz.data() : nullptr, threads);
    const double* outputs[3] = {fx.data(), fy.data(), fz.data()};
    gather(family_, position_, outputs, flux);
}

PfftFlux::PfftFlux(std::shared_ptr<const PfftOperator> op, std::vector<std::int8_t> family,
                   std::vector<std::int64_t> position, std::vector<std::int64_t> family_ids)
    : op_(std::move(op)), family_(std::move(family)), position_(std::move(position)), family_ids_(std::move(family_ids)) {
    if (family_ids_.size() != 3U) {
        throw InvalidInput("family_ids must name the x, y and z families");
    }
    sizes_.assign(3, 0);
    for (size_t f = 0; f < 3; ++f) {
        if (family_ids_[f] >= 0) {
            sizes_[f] = op_->planes(family_ids_[f]) * op_->branches(family_ids_[f]);
        }
    }
    check_positions(family_, position_, sizes_.data());
}

void PfftFlux::apply(const double* const currents, double* const flux, const int threads) const {
    std::vector<double> in[3];
    std::vector<double> out[3];
    double* inputs[3];
    const double* outputs[3];
    for (size_t f = 0; f < 3; ++f) {
        in[f].assign(static_cast<size_t>(sizes_[f]), 0.0);
        out[f].assign(static_cast<size_t>(sizes_[f]), 0.0);
        inputs[f] = in[f].data();
        outputs[f] = out[f].data();
    }
    scatter(family_, position_, currents, inputs);
    for (size_t f = 0; f < 3; ++f) {
        if (family_ids_[f] >= 0 && sizes_[f] > 0) {
            op_->apply(family_ids_[f], in[f].data(), out[f].data(), threads);
        }
    }
    gather(family_, position_, outputs, flux);
}

// ------------------------------------------------------------------- solve

namespace {

double norm2(const Complex* const x, const std::int64_t n) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        sum += std::norm(x[i]);
    }
    return std::sqrt(sum);
}

double norm2(const double* const x, const std::int64_t n) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

std::string format_complex(const Complex value) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "(%.6g%+.6gj)", value.real(), value.imag());
    return buffer;
}

// Components labelled in order of their lowest node, as SciPy's
// connected_components labels them.
std::vector<std::int64_t> component_labels(const SheetProblem& p, std::int64_t& count) {
    std::vector<std::int64_t> parent(static_cast<size_t>(p.node_count));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&parent](std::int64_t a) {
        while (parent[static_cast<size_t>(a)] != a) {
            parent[static_cast<size_t>(a)] = parent[static_cast<size_t>(parent[static_cast<size_t>(a)])];
            a = parent[static_cast<size_t>(a)];
        }
        return a;
    };
    for (std::int64_t b = 0; b < p.branch_count; ++b) {
        const std::int64_t ra = find(p.left[b]);
        const std::int64_t rb = find(p.right[b]);
        if (ra != rb) {
            parent[static_cast<size_t>(std::max(ra, rb))] = std::min(ra, rb);
        }
    }
    std::vector<std::int64_t> label_of_root(static_cast<size_t>(p.node_count), -1);
    std::vector<std::int64_t> labels(static_cast<size_t>(p.node_count));
    count = 0;
    for (std::int64_t node = 0; node < p.node_count; ++node) {
        const std::int64_t root = find(node);
        if (label_of_root[static_cast<size_t>(root)] < 0) {
            label_of_root[static_cast<size_t>(root)] = count++;
        }
        labels[static_cast<size_t>(node)] = label_of_root[static_cast<size_t>(root)];
    }
    return labels;
}

struct Reduction {
    std::vector<bool> kept_node;          // node lies in a driven component
    std::vector<std::int64_t> unknown;    // node -> unknown index, -1 if grounded or dropped
    std::int64_t unknowns{0};
    std::vector<std::int64_t> active;     // active branch -> mesh branch
    std::vector<std::int64_t> row_left;   // active branch -> unknown of its left node (or -1)
    std::vector<std::int64_t> row_right;  // ... of its right node (or -1)
    std::int64_t grounded{0};
    std::int64_t dropped{0};
};

Reduction reduce(const SheetProblem& p) {
    std::int64_t components = 0;
    const std::vector<std::int64_t> labels = component_labels(p, components);
    std::vector<double> magnitude(static_cast<size_t>(components), 0.0);
    std::vector<Complex> total(static_cast<size_t>(components), Complex(0.0, 0.0));
    for (std::int64_t node = 0; node < p.node_count; ++node) {
        const auto label = static_cast<size_t>(labels[static_cast<size_t>(node)]);
        magnitude[label] += std::abs(p.injection[node]);
        total[label] += p.injection[node];
    }
    std::vector<bool> carried(static_cast<size_t>(components), false);
    bool any = false;
    for (std::int64_t c = 0; c < components; ++c) {
        carried[static_cast<size_t>(c)] = magnitude[static_cast<size_t>(c)] > 0.0;
        any = any || carried[static_cast<size_t>(c)];
    }
    if (!any) {
        throw InvalidInput("no connected component of the conductor carries a terminal");
    }
    for (std::int64_t c = 0; c < components; ++c) {
        if (carried[static_cast<size_t>(c)] &&
            std::abs(total[static_cast<size_t>(c)]) > 1e-9 * std::max(1.0, magnitude[static_cast<size_t>(c)])) {
            throw InvalidInput("connected component " + std::to_string(c) + " is given a net " +
                               format_complex(total[static_cast<size_t>(c)]) +
                               " A; each isolated component has to close its own current");
        }
    }
    Reduction r;
    r.kept_node.assign(static_cast<size_t>(p.node_count), false);
    std::vector<bool> keep(static_cast<size_t>(p.node_count), false);
    std::vector<bool> grounded_component(static_cast<size_t>(components), false);
    bool first_grounded = true;
    for (std::int64_t node = 0; node < p.node_count; ++node) {
        const auto label = static_cast<size_t>(labels[static_cast<size_t>(node)]);
        if (!carried[label]) {
            ++r.dropped;
            continue;
        }
        r.kept_node[static_cast<size_t>(node)] = true;
        if (!grounded_component[label]) {
            grounded_component[label] = true;  // the first node of each driven component
            if (first_grounded) {
                r.grounded = node;
                first_grounded = false;
            }
            continue;
        }
        keep[static_cast<size_t>(node)] = true;
    }
    // The grounded node of the lowest-labelled driven component, as the
    // NumPy loop over sorted components picks it.
    for (std::int64_t c = 0; c < components; ++c) {
        if (carried[static_cast<size_t>(c)]) {
            for (std::int64_t node = 0; node < p.node_count; ++node) {
                if (labels[static_cast<size_t>(node)] == c) {
                    r.grounded = node;
                    break;
                }
            }
            break;
        }
    }
    r.unknown.assign(static_cast<size_t>(p.node_count), -1);
    for (std::int64_t node = 0; node < p.node_count; ++node) {
        if (keep[static_cast<size_t>(node)]) {
            r.unknown[static_cast<size_t>(node)] = r.unknowns++;
        }
    }
    for (std::int64_t b = 0; b < p.branch_count; ++b) {
        if (r.kept_node[static_cast<size_t>(p.left[b])] && r.kept_node[static_cast<size_t>(p.right[b])]) {
            r.active.push_back(b);
            r.row_left.push_back(r.unknown[static_cast<size_t>(p.left[b])]);
            r.row_right.push_back(r.unknown[static_cast<size_t>(p.right[b])]);
        }
    }
    return r;
}

// A^T y over active branches: +y at the left node's unknown, -y at the right's.
template <typename T>
void incidence_transposed(const Reduction& r, const T* const y, T* const out) {
    std::fill(out, out + r.unknowns, T(0));
    for (size_t b = 0; b < r.active.size(); ++b) {
        if (r.row_left[b] >= 0) {
            out[r.row_left[b]] += y[b];
        }
        if (r.row_right[b] >= 0) {
            out[r.row_right[b]] -= y[b];
        }
    }
}

// A v: the drop along each active branch.
template <typename T>
void incidence(const Reduction& r, const T* const v, T* const out) {
    for (size_t b = 0; b < r.active.size(); ++b) {
        T value(0);
        if (r.row_left[b] >= 0) {
            value += v[r.row_left[b]];
        }
        if (r.row_right[b] >= 0) {
            value -= v[r.row_right[b]];
        }
        out[b] = value;
    }
}

// A^T diag(1/d) A over the unknowns.
template <typename T>
linalg::CscMatrixT<T> nodal_matrix(const Reduction& r, const std::vector<T>& inverse_d) {
    linalg::Triplets<T> t;
    t.reserve(4 * r.active.size());
    for (size_t b = 0; b < r.active.size(); ++b) {
        const std::int64_t u = r.row_left[b];
        const std::int64_t v = r.row_right[b];
        const T y = inverse_d[b];
        if (u >= 0) {
            t.add(u, u, y);
        }
        if (v >= 0) {
            t.add(v, v, y);
        }
        if (u >= 0 && v >= 0) {
            t.add(u, v, -y);
            t.add(v, u, -y);
        }
    }
    return linalg::csc_from_triplets(r.unknowns, r.unknowns, t);
}

// R + j w L_near over the active branches, as triplets in active numbering.
linalg::Triplets<Complex> near_impedance(const SheetProblem& p, const Reduction& r, const double omega) {
    if (p.near == nullptr) {
        throw InvalidInput("the near and block preconditioners need the near-field inductance");
    }
    std::vector<std::int64_t> active_index(static_cast<size_t>(p.branch_count), -1);
    for (size_t b = 0; b < r.active.size(); ++b) {
        active_index[static_cast<size_t>(r.active[b])] = static_cast<std::int64_t>(b);
    }
    linalg::Triplets<Complex> t;
    for (size_t b = 0; b < r.active.size(); ++b) {
        t.add(static_cast<std::int64_t>(b), static_cast<std::int64_t>(b), Complex(p.resistance[r.active[b]], 0.0));
    }
    for (size_t b = 0; b < r.active.size(); ++b) {
        const std::int64_t row = r.active[b];
        for (std::int64_t k = p.near->indptr[static_cast<size_t>(row)]; k < p.near->indptr[static_cast<size_t>(row) + 1U];
             ++k) {
            const std::int64_t column = active_index[static_cast<size_t>(p.near->indices[static_cast<size_t>(k)])];
            if (column >= 0) {
                t.add(static_cast<std::int64_t>(b), column, Complex(0.0, omega * p.near->data[static_cast<size_t>(k)]));
            }
        }
    }
    return t;
}

using Apply = std::function<void(const Complex*, Complex*)>;

Apply build_preconditioner(const SheetProblem& p, const Reduction& r, const std::string& kind, const double omega,
                           std::vector<std::shared_ptr<void>>& keep_alive) {
    const auto branches = static_cast<std::int64_t>(r.active.size());
    const std::int64_t unknowns = r.unknowns;
    if (kind == "near") {
        linalg::Triplets<Complex> t = near_impedance(p, r, omega);
        for (std::int64_t b = 0; b < branches; ++b) {
            if (r.row_left[static_cast<size_t>(b)] >= 0) {
                t.add(b, branches + r.row_left[static_cast<size_t>(b)], Complex(-1.0, 0.0));
                t.add(branches + r.row_left[static_cast<size_t>(b)], b, Complex(1.0, 0.0));
            }
            if (r.row_right[static_cast<size_t>(b)] >= 0) {
                t.add(b, branches + r.row_right[static_cast<size_t>(b)], Complex(1.0, 0.0));
                t.add(branches + r.row_right[static_cast<size_t>(b)], b, Complex(-1.0, 0.0));
            }
        }
        auto lu = std::make_shared<linalg::ComplexSparseLU>(
            linalg::csc_from_triplets(branches + unknowns, branches + unknowns, t));
        keep_alive.push_back(lu);
        const std::int64_t size = branches + unknowns;
        return [lu, size](const Complex* in, Complex* out) {
            std::copy(in, in + size, out);
            lu->solve(out, 1);
        };
    }
    std::vector<Complex> diagonal(static_cast<size_t>(branches));
    std::vector<Complex> inverse(static_cast<size_t>(branches));
    if (kind == "block" || kind == "diagonal") {
        if (p.self_inductance == nullptr) {
            throw InvalidInput("the block and diagonal preconditioners need the branch self inductance");
        }
        for (std::int64_t b = 0; b < branches; ++b) {
            const std::int64_t m = r.active[static_cast<size_t>(b)];
            diagonal[static_cast<size_t>(b)] = Complex(p.resistance[m], omega * p.self_inductance[m]);
            inverse[static_cast<size_t>(b)] = 1.0 / diagonal[static_cast<size_t>(b)];
        }
    }
    if (kind == "block") {
        auto z = std::make_shared<linalg::ComplexSparseLU>(
            linalg::csc_from_triplets(branches, branches, near_impedance(p, r, omega)),
            linalg::ColumnOrdering::mmd_at_plus_a);
        auto s = std::make_shared<linalg::ComplexSparseLU>(nodal_matrix(r, inverse),
                                                           linalg::ColumnOrdering::mmd_at_plus_a);
        keep_alive.push_back(z);
        keep_alive.push_back(s);
        const Reduction* rr = &r;
        return [z, s, rr, branches, unknowns](const Complex* in, Complex* out) {
            // [[Z, -A], [A^T, 0]] [x; y] = [r1; r2]:
            // x = Z^-1 (r1 + A y),  A^T Z^-1 A y = r2 - A^T Z^-1 r1.
            std::vector<Complex> zr(in, in + branches);
            z->solve(zr.data(), 1);
            std::vector<Complex> node(static_cast<size_t>(unknowns));
            incidence_transposed(*rr, zr.data(), node.data());
            for (std::int64_t i = 0; i < unknowns; ++i) {
                node[static_cast<size_t>(i)] = in[branches + i] - node[static_cast<size_t>(i)];
            }
            s->solve(node.data(), 1);
            std::vector<Complex> drop(static_cast<size_t>(branches));
            incidence(*rr, node.data(), drop.data());
            for (std::int64_t b = 0; b < branches; ++b) {
                out[b] = in[b] + drop[static_cast<size_t>(b)];
            }
            z->solve(out, 1);
            std::copy(node.begin(), node.end(), out + branches);
        };
    }
    if (kind == "diagonal") {
        auto s = std::make_shared<linalg::ComplexSparseLU>(nodal_matrix(r, inverse));
        keep_alive.push_back(s);
        const Reduction* rr = &r;
        auto inv = std::make_shared<std::vector<Complex>>(std::move(inverse));
        auto diag = std::make_shared<std::vector<Complex>>(std::move(diagonal));
        return [s, rr, inv, diag, branches, unknowns](const Complex* in, Complex* out) {
            // Block factorisation of [[D, -A], [A^T, 0]].
            std::vector<Complex> scaled(static_cast<size_t>(branches));
            for (std::int64_t b = 0; b < branches; ++b) {
                scaled[static_cast<size_t>(b)] = in[b] / (*diag)[static_cast<size_t>(b)];
            }
            std::vector<Complex> node(static_cast<size_t>(unknowns));
            incidence_transposed(*rr, scaled.data(), node.data());
            for (std::int64_t i = 0; i < unknowns; ++i) {
                node[static_cast<size_t>(i)] = in[branches + i] + node[static_cast<size_t>(i)];
            }
            s->solve(node.data(), 1);
            std::vector<Complex> drop(static_cast<size_t>(branches));
            incidence(*rr, node.data(), drop.data());
            for (std::int64_t b = 0; b < branches; ++b) {
                out[b] = (in[b] + drop[static_cast<size_t>(b)]) / (*diag)[static_cast<size_t>(b)];
            }
            std::copy(node.begin(), node.end(), out + branches);
        };
    }
    throw InvalidInput("preconditioner must be 'auto', 'near', 'block' or 'diagonal'");
}

}  // namespace

SheetResult solve_sheet(const SheetProblem& p, const int threads) {
    if (p.node_count < 1 || p.branch_count < 0 || p.left == nullptr || p.right == nullptr ||
        p.resistance == nullptr || p.injection == nullptr) {
        throw InvalidInput("incomplete sheet problem");
    }
    for (std::int64_t b = 0; b < p.branch_count; ++b) {
        if (p.left[b] < 0 || p.left[b] >= p.node_count || p.right[b] < 0 || p.right[b] >= p.node_count) {
            throw InvalidInput("branch endpoint outside the mesh nodes");
        }
    }
    const Reduction r = reduce(p);
    const auto branches = static_cast<std::int64_t>(r.active.size());
    const std::int64_t unknowns = r.unknowns;

    SheetResult result;
    result.grounded_node = r.grounded;
    result.undriven_nodes = r.dropped;
    result.node_voltage.assign(static_cast<size_t>(p.node_count), Complex(0.0, 0.0));
    result.branch_current.assign(static_cast<size_t>(p.branch_count), Complex(0.0, 0.0));
    std::vector<Complex> injected(static_cast<size_t>(unknowns));
    for (std::int64_t node = 0; node < p.node_count; ++node) {
        if (r.unknown[static_cast<size_t>(node)] >= 0) {
            injected[static_cast<size_t>(r.unknown[static_cast<size_t>(node)])] = p.injection[node];
        }
    }

    if (p.frequency_hz == 0.0) {
        std::vector<double> admittance(static_cast<size_t>(branches));
        for (std::int64_t b = 0; b < branches; ++b) {
            admittance[static_cast<size_t>(b)] = 1.0 / p.resistance[r.active[static_cast<size_t>(b)]];
        }
        const linalg::SparseLU lu(nodal_matrix(r, admittance));
        std::vector<double> voltage(static_cast<size_t>(unknowns));
        for (std::int64_t i = 0; i < unknowns; ++i) {
            voltage[static_cast<size_t>(i)] = injected[static_cast<size_t>(i)].real();
        }
        lu.solve(voltage.data(), 1);
        std::vector<double> drop(static_cast<size_t>(branches));
        incidence(r, voltage.data(), drop.data());
        std::vector<double> current(static_cast<size_t>(branches));
        for (std::int64_t b = 0; b < branches; ++b) {
            current[static_cast<size_t>(b)] = admittance[static_cast<size_t>(b)] * drop[static_cast<size_t>(b)];
        }
        std::vector<double> balance(static_cast<size_t>(unknowns));
        incidence_transposed(r, current.data(), balance.data());
        std::vector<double> rhs(static_cast<size_t>(unknowns));
        for (std::int64_t i = 0; i < unknowns; ++i) {
            rhs[static_cast<size_t>(i)] = injected[static_cast<size_t>(i)].real();
            balance[static_cast<size_t>(i)] -= rhs[static_cast<size_t>(i)];
        }
        for (std::int64_t node = 0; node < p.node_count; ++node) {
            const std::int64_t u = r.unknown[static_cast<size_t>(node)];
            if (u >= 0) {
                result.node_voltage[static_cast<size_t>(node)] = voltage[static_cast<size_t>(u)];
            }
        }
        for (std::int64_t b = 0; b < branches; ++b) {
            result.branch_current[static_cast<size_t>(r.active[static_cast<size_t>(b)])] =
                current[static_cast<size_t>(b)];
        }
        result.residual = norm2(balance.data(), unknowns) / std::max(norm2(rhs.data(), unknowns), 1e-30);
        result.iterations = 1;
        result.converged = result.residual < 1e-8;
        result.preconditioner = "direct";
        return result;
    }

    if (p.flux == nullptr || p.flux->branch_count() != p.branch_count) {
        throw InvalidInput("the AC solve needs a flux operator over the mesh branches");
    }
    const double omega = 2.0 * kPi * p.frequency_hz;
    const std::int64_t size = branches + unknowns;
    std::string kind = p.preconditioner;
    if (kind == "auto") {
        kind = size >= p.auto_block_from_unknowns ? "block" : "near";
    }
    result.preconditioner = kind;

    std::vector<double> full_real(static_cast<size_t>(p.branch_count), 0.0);
    std::vector<double> flux_real(static_cast<size_t>(p.branch_count));
    std::vector<double> flux_imag(static_cast<size_t>(p.branch_count));
    std::vector<Complex> drop(static_cast<size_t>(branches));
    auto saddle = [&](const Complex* in, Complex* out) {
        // Z I = R I + j w (L Re I + j L Im I), the flux applied to each part.
        for (std::int64_t b = 0; b < branches; ++b) {
            full_real[static_cast<size_t>(r.active[static_cast<size_t>(b)])] = in[b].real();
        }
        p.flux->apply(full_real.data(), flux_real.data(), threads);
        for (std::int64_t b = 0; b < branches; ++b) {
            full_real[static_cast<size_t>(r.active[static_cast<size_t>(b)])] = in[b].imag();
        }
        p.flux->apply(full_real.data(), flux_imag.data(), threads);
        incidence(r, in + branches, drop.data());
        for (std::int64_t b = 0; b < branches; ++b) {
            const std::int64_t m = r.active[static_cast<size_t>(b)];
            const Complex flux(flux_real[static_cast<size_t>(m)], flux_imag[static_cast<size_t>(m)]);
            out[b] = p.resistance[m] * in[b] + Complex(0.0, omega) * flux - drop[static_cast<size_t>(b)];
        }
        incidence_transposed(r, in, out + branches);
    };

    std::vector<std::shared_ptr<void>> keep_alive;
    const Apply precondition = build_preconditioner(p, r, kind, omega, keep_alive);

    std::vector<Complex> rhs(static_cast<size_t>(size), Complex(0.0, 0.0));
    std::copy(injected.begin(), injected.end(), rhs.begin() + branches);
    std::vector<Complex> solution(static_cast<size_t>(size), Complex(0.0, 0.0));
    const linalg::GmresResult gmres =
        linalg::gmres(saddle, precondition, rhs, solution, p.tolerance, 0.0, p.restart, p.max_iterations);

    std::vector<Complex> check(static_cast<size_t>(size));
    saddle(solution.data(), check.data());
    for (std::int64_t i = 0; i < size; ++i) {
        check[static_cast<size_t>(i)] -= rhs[static_cast<size_t>(i)];
    }
    result.residual = norm2(check.data(), size) / std::max(norm2(rhs.data(), size), 1e-30);
    result.iterations = gmres.inner_iterations;
    result.converged = gmres.info == 0 && result.residual < 1e-6;
    for (std::int64_t b = 0; b < branches; ++b) {
        result.branch_current[static_cast<size_t>(r.active[static_cast<size_t>(b)])] =
            solution[static_cast<size_t>(b)];
    }
    for (std::int64_t node = 0; node < p.node_count; ++node) {
        const std::int64_t u = r.unknown[static_cast<size_t>(node)];
        if (u >= 0) {
            result.node_voltage[static_cast<size_t>(node)] = solution[static_cast<size_t>(branches + u)];
        }
    }
    return result;
}

}  // namespace pcbcore::sheet
