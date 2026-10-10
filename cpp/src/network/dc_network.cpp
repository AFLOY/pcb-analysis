#include "pcbcore/network/dc_network.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "pcbcore/errors.hpp"

namespace pcbcore::network {

namespace {

using std::size_t;

// Row of each node in the reduced system; -1 for the reference.
std::vector<std::int64_t> unknown_index(const std::int64_t node_count, const std::int64_t reference) {
    if (reference < 0 || reference >= node_count) {
        throw InvalidInput("reference node outside the network");
    }
    std::vector<std::int64_t> index(static_cast<size_t>(node_count));
    for (std::int64_t node = 0; node < node_count; ++node) {
        index[static_cast<size_t>(node)] = node - (node > reference ? 1 : 0);
    }
    index[static_cast<size_t>(reference)] = -1;
    return index;
}

double norm2(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

}  // namespace

void ConductanceNetworkView::validate() const {
    if (node_count < 0 || branch_count < 0) {
        throw InvalidInput("negative node or branch count");
    }
    for (std::int64_t k = 0; k < branch_count; ++k) {
        if (left[k] < 0 || left[k] >= node_count || right[k] < 0 || right[k] >= node_count) {
            throw InvalidInput("branch endpoint outside 0..node_count-1");
        }
    }
}

linalg::CscMatrix reduced_laplacian(const ConductanceNetworkView& network, const std::int64_t reference) {
    network.validate();
    const std::vector<std::int64_t> unknown = unknown_index(network.node_count, reference);
    const std::int64_t count = network.node_count - 1;

    // Entries in a fixed order: both off-diagonal entries of each interior
    // branch in branch order, then the diagonal accumulated endpoint by
    // endpoint in branch order.
    std::vector<double> diagonal(static_cast<size_t>(count), 0.0);
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> values;
    rows.reserve(static_cast<size_t>(2 * network.branch_count + count));
    cols.reserve(rows.capacity());
    values.reserve(rows.capacity());
    for (std::int64_t k = 0; k < network.branch_count; ++k) {
        const std::int64_t a = unknown[static_cast<size_t>(network.left[k])];
        const std::int64_t b = unknown[static_cast<size_t>(network.right[k])];
        const double g = network.conductance[k];
        if (a >= 0) {
            diagonal[static_cast<size_t>(a)] += g;
        }
        if (b >= 0) {
            diagonal[static_cast<size_t>(b)] += g;
        }
        if (a >= 0 && b >= 0) {
            rows.push_back(a);
            cols.push_back(b);
            values.push_back(-g);
            rows.push_back(b);
            cols.push_back(a);
            values.push_back(-g);
        }
    }
    for (std::int64_t i = 0; i < count; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        values.push_back(diagonal[static_cast<size_t>(i)]);
    }

    // Stable bucket by column, then a stable sort by row inside each column,
    // and duplicates summed in the order they were listed.
    linalg::CscMatrix matrix;
    matrix.rows = count;
    matrix.cols = count;
    std::vector<std::int64_t> start(static_cast<size_t>(count) + 1U, 0);
    for (const std::int64_t col : cols) {
        ++start[static_cast<size_t>(col) + 1U];
    }
    std::partial_sum(start.begin(), start.end(), start.begin());
    std::vector<std::int64_t> order(values.size());
    {
        std::vector<std::int64_t> next(start.begin(), start.end() - 1);
        for (size_t entry = 0; entry < values.size(); ++entry) {
            order[static_cast<size_t>(next[static_cast<size_t>(cols[entry])]++)] =
                static_cast<std::int64_t>(entry);
        }
    }
    matrix.column_start.assign(static_cast<size_t>(count) + 1U, 0);
    matrix.row_index.reserve(values.size());
    matrix.value.reserve(values.size());
    for (std::int64_t col = 0; col < count; ++col) {
        const auto first = order.begin() + start[static_cast<size_t>(col)];
        const auto last = order.begin() + start[static_cast<size_t>(col) + 1U];
        std::stable_sort(first, last, [&rows](const std::int64_t x, const std::int64_t y) {
            return rows[static_cast<size_t>(x)] < rows[static_cast<size_t>(y)];
        });
        for (auto it = first; it != last; ++it) {
            const std::int64_t row = rows[static_cast<size_t>(*it)];
            const double value = values[static_cast<size_t>(*it)];
            if (!matrix.row_index.empty() &&
                static_cast<std::int64_t>(matrix.row_index.size()) > matrix.column_start[static_cast<size_t>(col)] &&
                matrix.row_index.back() == row) {
                matrix.value.back() += value;
            } else {
                matrix.row_index.push_back(row);
                matrix.value.push_back(value);
            }
        }
        matrix.column_start[static_cast<size_t>(col) + 1U] = static_cast<std::int64_t>(matrix.value.size());
    }
    return matrix;
}

DCNetworkResult solve_conductance_network(const ConductanceNetworkView& network,
                                          const std::int64_t reference,
                                          const double* const injection,
                                          const double* const weights,
                                          const std::int64_t objectives) {
    if (objectives < 0 || (objectives > 0 && weights == nullptr)) {
        throw InvalidInput("objective weights are missing");
    }
    const linalg::CscMatrix matrix = reduced_laplacian(network, reference);
    const std::vector<std::int64_t> unknown = unknown_index(network.node_count, reference);
    const std::int64_t nodes = network.node_count;
    const std::int64_t count = nodes - 1;
    const std::int64_t columns = 1 + objectives;

    // Column-major block: the injection, then one adjoint right-hand side per objective.
    std::vector<double> block(static_cast<size_t>(count * columns), 0.0);
    for (std::int64_t node = 0; node < nodes; ++node) {
        const std::int64_t row = unknown[static_cast<size_t>(node)];
        if (row < 0) {
            continue;
        }
        block[static_cast<size_t>(row)] = injection[node];
        for (std::int64_t j = 0; j < objectives; ++j) {
            block[static_cast<size_t>((j + 1) * count + row)] = weights[j * nodes + node];
        }
    }
    const std::vector<double> rhs(block.begin(), block.begin() + count);

    DCNetworkResult result;
    {
        const linalg::SparseLU lu(matrix);
        result.singular = lu.singular();
        lu.solve(block.data(), columns);
    }
    result.voltage_unknowns.assign(block.begin(), block.begin() + count);

    std::vector<double> residual(static_cast<size_t>(count), 0.0);
    matrix.multiply(result.voltage_unknowns.data(), residual.data());
    for (std::int64_t i = 0; i < count; ++i) {
        residual[static_cast<size_t>(i)] -= rhs[static_cast<size_t>(i)];
    }
    result.relative_residual = norm2(residual) / std::max(norm2(rhs), 1e-30);

    auto full = [&](const double* const values, double* const out) {
        for (std::int64_t node = 0; node < nodes; ++node) {
            const std::int64_t row = unknown[static_cast<size_t>(node)];
            out[node] = row >= 0 ? values[row] : 0.0;
        }
    };
    result.node_voltage.assign(static_cast<size_t>(nodes), 0.0);
    full(result.voltage_unknowns.data(), result.node_voltage.data());
    result.adjoint_voltage.assign(static_cast<size_t>(objectives * nodes), 0.0);
    for (std::int64_t j = 0; j < objectives; ++j) {
        full(block.data() + (j + 1) * count, result.adjoint_voltage.data() + j * nodes);
    }

    const std::int64_t branches = network.branch_count;
    result.edge_current.assign(static_cast<size_t>(branches), 0.0);
    result.node_current.assign(static_cast<size_t>(nodes), 0.0);
    double loss = 0.0;  // a running sum: reproducible branch order for branch order
    for (std::int64_t k = 0; k < branches; ++k) {
        const std::int64_t a = network.left[k];
        const std::int64_t b = network.right[k];
        const double g = network.conductance[k];
        const double current =
            g * (result.node_voltage[static_cast<size_t>(a)] - result.node_voltage[static_cast<size_t>(b)]);
        result.edge_current[static_cast<size_t>(k)] = current;
        result.node_current[static_cast<size_t>(a)] += current;
        result.node_current[static_cast<size_t>(b)] += -current;
        loss += current * current / g;
    }
    result.loss_w = loss;
    return result;
}

BranchSensitivity split_branch_sensitivity(const ConductanceNetworkView& network,
                                           const double* const branch_product,
                                           const bool* const in_plane) {
    network.validate();
    const std::int64_t branches = network.branch_count;
    BranchSensitivity out;
    out.node_sensitivity.assign(static_cast<size_t>(network.node_count), 0.0);
    out.branch_sensitivity.assign(static_cast<size_t>(branches), 0.0);
    std::vector<double> halves(static_cast<size_t>(branches), 0.0);
    for (std::int64_t k = 0; k < branches; ++k) {
        const double sensitivity = network.conductance[k] * branch_product[k];
        out.branch_sensitivity[static_cast<size_t>(k)] = sensitivity;
        if (in_plane[k]) {
            halves[static_cast<size_t>(k)] = network.conductance[k] * branch_product[k] * 0.5;
        } else {
            out.vertical_total += sensitivity;
        }
    }
    // Every left endpoint first, then every right one, as the NumPy version adds them.
    for (std::int64_t k = 0; k < branches; ++k) {
        out.node_sensitivity[static_cast<size_t>(network.left[k])] += halves[static_cast<size_t>(k)];
    }
    for (std::int64_t k = 0; k < branches; ++k) {
        out.node_sensitivity[static_cast<size_t>(network.right[k])] += halves[static_cast<size_t>(k)];
    }
    return out;
}

}  // namespace pcbcore::network
