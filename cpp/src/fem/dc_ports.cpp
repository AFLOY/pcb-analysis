#include "pcbcore/fem/dc_ports.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <thread>

#include "pcbcore/errors.hpp"

namespace pcbcore::fem {

namespace {

using Index = std::int64_t;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct UnitSolve {
    std::vector<double> potential;
    std::vector<double> currents;
    MpirResult result;
};

UnitSolve unit_solve(const LayeredDCSystem& system, const NodeGroupsView& ports, const Index port,
                     const double* const previous, const MpirConfig& config, const int team) {
    std::vector<double> excitation(static_cast<std::size_t>(ports.count), 0.0);
    excitation[port] = 1.0;
    const NodeGroupsView none{0, nullptr, nullptr};
    const std::vector<double> rhs = system.build_rhs(none, nullptr, ports, excitation.data(), team);
    UnitSolve out;
    out.potential = system.dirichlet_potential(ports, excitation.data());
    if (previous != nullptr) {
        const auto& free = system.free_nodes();
        for (Index n = 0; n < system.size(); ++n) {
            if (free[n] != 0U) {
                out.potential[n] = previous[n];
            }
        }
    }
    out.result = system.solve(rhs.data(), out.potential.data(), config, team);
    out.currents = system.group_currents(out.potential.data(), ports, team);
    return out;
}

}  // namespace

DCPortBasisResult dc_port_basis(const LayeredDCSystem& system, const NodeGroupsView& ports, const std::int64_t reference,
                                const double* const initial_unit_voltage, const MpirConfig& config, const int width,
                                const int team) {
    const Index n = ports.count;
    if (n < 2 || reference < 0 || reference >= n) {
        throw InvalidInput("a port basis needs two ports and a reference among them");
    }
    std::vector<Index> driven;
    for (Index k = 0; k < n; ++k) {
        if (k != reference) {
            driven.push_back(k);
        }
    }
    const Index m = n - 1;
    const Index nodes = system.size();

    // Unit solves: worker w takes columns w, w + width, ...; each owns its
    // vectors and the system is read-only, so the split changes nothing.
    std::vector<UnitSolve> solves(static_cast<std::size_t>(m));
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(m));
    const int pool = static_cast<int>(std::max<Index>(1, std::min<Index>(width, m)));
    const auto work = [&](const int worker) {
        for (Index column = worker; column < m; column += pool) {
            try {
                const double* previous =
                    initial_unit_voltage == nullptr ? nullptr : initial_unit_voltage + column * nodes;
                solves[column] = unit_solve(system, ports, driven[column], previous, config, std::max(1, team));
            } catch (...) {
                errors[column] = std::current_exception();
            }
        }
    };
    if (pool == 1) {
        work(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(pool));
        for (int worker = 0; worker < pool; ++worker) {
            threads.emplace_back(work, worker);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }

    DCPortBasisResult out;
    out.ports = n;
    // The driven block is what the solves measured, symmetrised against
    // solver round-off; the reference row and column follow from KCL exactly.
    RowMatrix block(m, m);
    for (Index j = 0; j < m; ++j) {
        for (Index k = 0; k < m; ++k) {
            block(j, k) = solves[k].currents[driven[j]];
        }
    }
    RowMatrix symmetric(m, m);
    for (Index j = 0; j < m; ++j) {
        for (Index k = 0; k < m; ++k) {
            symmetric(j, k) = 0.5 * (block(j, k) + block(k, j));
        }
    }
    out.conductance.assign(static_cast<std::size_t>(n * n), 0.0);
    double total = 0.0;
    for (Index j = 0; j < m; ++j) {
        double row = 0.0;
        double column = 0.0;
        for (Index k = 0; k < m; ++k) {
            out.conductance[driven[j] * n + driven[k]] = symmetric(j, k);
            row += symmetric(j, k);
            column += symmetric(k, j);
            total += symmetric(j, k);
        }
        out.conductance[driven[j] * n + reference] = -row;
        out.conductance[reference * n + driven[j]] = -column;
    }
    out.conductance[reference * n + reference] = total;

    Eigen::PartialPivLU<RowMatrix> lu(symmetric);
    if (!(std::abs(lu.determinant()) > 0.0)) {
        throw Singular("the driven conductance block is singular");
    }
    const RowMatrix resistance = lu.inverse();

    out.unit_voltage.assign(static_cast<std::size_t>(m * nodes), 0.0);
    for (Index column = 0; column < m; ++column) {
        std::copy(solves[column].potential.begin(), solves[column].potential.end(),
                  out.unit_voltage.begin() + column * nodes);
    }
    // phi_k (1 A into driven port k) = sum_j R_jk (unit voltage field of port j).
    out.unit_current.assign(static_cast<std::size_t>(m * nodes), 0.0);
    for (Index k = 0; k < m; ++k) {
        double* const target = out.unit_current.data() + k * nodes;
        for (Index j = 0; j < m; ++j) {
            const double weight = resistance(j, k);
            const double* const source = out.unit_voltage.data() + j * nodes;
            for (Index node = 0; node < nodes; ++node) {
                target[node] += weight * source[node];
            }
        }
    }
    for (auto& solve : solves) {
        out.solves.push_back(std::move(solve.result));
    }
    return out;
}

CorrelationModes correlation_modes(const double* const reduced, const std::int64_t driven, const double scale,
                                   const double* const unit_current, const std::int64_t nodes) {
    RowMatrix matrix(driven, driven);
    for (Index j = 0; j < driven; ++j) {
        for (Index k = 0; k < driven; ++k) {
            matrix(j, k) = reduced[j * driven + k];
        }
    }
    Eigen::SelfAdjointEigenSolver<RowMatrix> solver(matrix);
    if (solver.info() != Eigen::Success) {
        throw Singular("the correlation eigen-decomposition did not converge");
    }
    const auto& values = solver.eigenvalues();
    const auto& vectors = solver.eigenvectors();
    CorrelationModes out;
    if (driven > 0 && values.minCoeff() < -1.0e-10 * scale) {
        throw InvalidInput("correlation must be positive semi-definite");
    }
    for (Index mode = 0; mode < driven; ++mode) {
        if (!(values(mode) > 1.0e-14 * scale)) {
            continue;
        }
        out.weights.push_back(values(mode));
        const std::size_t base = out.modes.size();
        out.modes.resize(base + static_cast<std::size_t>(nodes), 0.0);
        double* const target = out.modes.data() + base;
        for (Index k = 0; k < driven; ++k) {
            const double weight = vectors(k, mode);
            const double* const source = unit_current + k * nodes;
            for (Index node = 0; node < nodes; ++node) {
                target[node] += weight * source[node];
            }
        }
    }
    return out;
}

ModalLoss modal_loss(const LayeredDCSystem& system, const CorrelationModes& modes, const int threads) {
    const Index nodes = system.size();
    ModalLoss out;
    out.element.assign(static_cast<std::size_t>(system.layers()) * system.rows() * system.cols(), 0.0);
    out.via.assign(static_cast<std::size_t>(system.via_count()), 0.0);
    for (std::size_t mode = 0; mode < modes.weights.size(); ++mode) {
        const double weight = modes.weights[mode];
        const double* const potential = modes.modes.data() + mode * static_cast<std::size_t>(nodes);
        const std::vector<double> element = system.element_joule_loss(potential, threads);
        const std::vector<double> via = system.via_joule_loss(potential);
        for (std::size_t e = 0; e < element.size(); ++e) {
            out.element[e] += weight * element[e];
        }
        for (std::size_t k = 0; k < via.size(); ++k) {
            out.via[k] += weight * via[k];
        }
    }
    return out;
}

std::vector<double> modal_rms_current_density(const LayeredDCSystem& system, const CorrelationModes& modes,
                                              const int threads) {
    const Index nodes = system.size();
    const auto& conductivity = system.conductivity();
    std::vector<double> square(conductivity.size(), 0.0);
    for (std::size_t mode = 0; mode < modes.weights.size(); ++mode) {
        const double weight = modes.weights[mode];
        const double* const potential = modes.modes.data() + mode * static_cast<std::size_t>(nodes);
        const std::vector<double> field = system.element_electric_field(potential, threads);
        for (std::size_t e = 0; e < square.size(); ++e) {
            const double jx = conductivity[e] * field[2 * e];
            const double jy = conductivity[e] * field[2 * e + 1];
            square[e] += weight * (jx * jx + jy * jy);
        }
    }
    for (double& value : square) {
        value = std::sqrt(value);
    }
    return square;
}

}  // namespace pcbcore::fem
