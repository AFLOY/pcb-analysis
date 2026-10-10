#include "pcbcore/sheet/mesh.hpp"

#include <cstddef>

#include "pcbcore/errors.hpp"

namespace pcbcore::sheet {

namespace {

using std::size_t;

}  // namespace

MeshTopology build_topology(const std::int64_t layers, const std::int64_t rows, const std::int64_t cols,
                            const std::uint8_t* const occupancy, const std::int64_t* const vias,
                            const std::int64_t via_count) {
    if (layers < 1 || rows < 1 || cols < 1) {
        throw InvalidInput("the mesh needs at least one layer, row and column");
    }
    MeshTopology mesh;
    mesh.layers = layers;
    mesh.rows = rows;
    mesh.cols = cols;
    const std::int64_t cells = layers * rows * cols;
    mesh.node_of.assign(static_cast<size_t>(cells), -1);
    auto at = [&](const std::int64_t l, const std::int64_t r, const std::int64_t c) { return (l * rows + r) * cols + c; };
    for (std::int64_t k = 0; k < cells; ++k) {
        if (occupancy[k] != 0U) {
            mesh.node_of[static_cast<size_t>(k)] = mesh.node_count++;
        }
    }
    if (mesh.node_count == 0) {
        throw InvalidInput("the conductor is empty");
    }
    auto copper = [&](const std::int64_t l, const std::int64_t r, const std::int64_t c) {
        return occupancy[at(l, r, c)] != 0U;
    };
    for (std::int64_t l = 0; l < layers; ++l) {
        for (std::int64_t r = 0; r < rows; ++r) {
            for (std::int64_t c = 0; c + 1 < cols; ++c) {
                if (copper(l, r, c) && copper(l, r, c + 1)) {
                    mesh.branch_x.insert(mesh.branch_x.end(), {l, r, c});
                    mesh.left.push_back(mesh.node_of[static_cast<size_t>(at(l, r, c))]);
                    mesh.right.push_back(mesh.node_of[static_cast<size_t>(at(l, r, c + 1))]);
                }
            }
        }
    }
    for (std::int64_t l = 0; l < layers; ++l) {
        for (std::int64_t r = 0; r + 1 < rows; ++r) {
            for (std::int64_t c = 0; c < cols; ++c) {
                if (copper(l, r, c) && copper(l, r + 1, c)) {
                    mesh.branch_y.insert(mesh.branch_y.end(), {l, r, c});
                    mesh.left.push_back(mesh.node_of[static_cast<size_t>(at(l, r, c))]);
                    mesh.right.push_back(mesh.node_of[static_cast<size_t>(at(l, r + 1, c))]);
                }
            }
        }
    }
    for (std::int64_t v = 0; v < via_count; ++v) {
        const std::int64_t lower = vias[4 * v];
        const std::int64_t upper = vias[4 * v + 1];
        const std::int64_t r = vias[4 * v + 2];
        const std::int64_t c = vias[4 * v + 3];
        if (lower < 0 || lower >= layers || upper < 0 || upper >= layers || r < 0 || r >= rows || c < 0 || c >= cols) {
            throw InvalidInput("a vertical branch lies outside the mesh");
        }
        if (copper(lower, r, c) && copper(upper, r, c)) {
            mesh.via_kept.push_back(v);
            mesh.left.push_back(mesh.node_of[static_cast<size_t>(at(lower, r, c))]);
            mesh.right.push_back(mesh.node_of[static_cast<size_t>(at(upper, r, c))]);
        }
    }
    return mesh;
}

std::vector<double> branch_resistance(const MeshTopology& mesh, const double* const sheet_resistance,
                                      const double* const pitch_x, const double* const pitch_y,
                                      const double* const via_resistance) {
    const size_t count_x = mesh.branch_x.size() / 3U;
    const size_t count_y = mesh.branch_y.size() / 3U;
    std::vector<double> values;
    values.reserve(count_x + count_y + mesh.via_kept.size());
    for (size_t b = 0; b < count_x; ++b) {
        const std::int64_t l = mesh.branch_x[3 * b];
        const std::int64_t r = mesh.branch_x[3 * b + 1];
        const std::int64_t c = mesh.branch_x[3 * b + 2];
        const double length = 0.5 * (pitch_x[c] + pitch_x[c + 1]);
        const double width = pitch_y[r];
        values.push_back(sheet_resistance[l] * (length / width));
    }
    for (size_t b = 0; b < count_y; ++b) {
        const std::int64_t l = mesh.branch_y[3 * b];
        const std::int64_t r = mesh.branch_y[3 * b + 1];
        const std::int64_t c = mesh.branch_y[3 * b + 2];
        const double length = 0.5 * (pitch_y[r] + pitch_y[r + 1]);
        const double width = pitch_x[c];
        values.push_back(sheet_resistance[l] * (length / width));
    }
    for (const std::int64_t v : mesh.via_kept) {
        values.push_back(via_resistance[v]);
    }
    return values;
}

std::vector<std::complex<double>> source_vector(const MeshTopology& mesh, const std::int64_t terminals,
                                                const std::int64_t* const layer, const std::int64_t* const cell_start,
                                                const std::int64_t* const cells,
                                                const std::complex<double>* const current,
                                                std::vector<std::int64_t>& usable) {
    std::vector<std::complex<double>> injected(static_cast<size_t>(mesh.node_count), std::complex<double>(0.0, 0.0));
    usable.assign(static_cast<size_t>(terminals), 0);
    std::vector<std::int64_t> nodes;
    for (std::int64_t t = 0; t < terminals; ++t) {
        nodes.clear();
        const std::int64_t l = layer[t];
        for (std::int64_t k = cell_start[t]; k < cell_start[t + 1]; ++k) {
            const std::int64_t r = cells[2 * k];
            const std::int64_t c = cells[2 * k + 1];
            if (l < 0 || l >= mesh.layers || r < 0 || r >= mesh.rows || c < 0 || c >= mesh.cols) {
                continue;
            }
            const std::int64_t node = mesh.node_of[static_cast<size_t>((l * mesh.rows + r) * mesh.cols + c)];
            if (node >= 0) {
                nodes.push_back(node);
            }
        }
        usable[static_cast<size_t>(t)] = static_cast<std::int64_t>(nodes.size());
        if (nodes.empty()) {
            continue;
        }
        const std::complex<double> share = current[t] / static_cast<double>(nodes.size());
        for (const std::int64_t node : nodes) {
            injected[static_cast<size_t>(node)] += share;
        }
    }
    return injected;
}

void cell_density(const MeshTopology& mesh, const std::complex<double>* const branch_current,
                  const double* const pitch_x, const double* const pitch_y, const double* const thickness_m,
                  std::complex<double>* const along_x, std::complex<double>* const along_y) {
    const size_t cells = static_cast<size_t>(mesh.layers * mesh.rows * mesh.cols);
    std::vector<std::complex<double>> sum_x(cells, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> sum_y(cells, std::complex<double>(0.0, 0.0));
    const size_t count_x = mesh.branch_x.size() / 3U;
    const size_t count_y = mesh.branch_y.size() / 3U;
    auto at = [&](const std::int64_t l, const std::int64_t r, const std::int64_t c) {
        return static_cast<size_t>((l * mesh.rows + r) * mesh.cols + c);
    };
    for (size_t b = 0; b < count_x; ++b) {
        const std::int64_t l = mesh.branch_x[3 * b];
        const std::int64_t r = mesh.branch_x[3 * b + 1];
        const std::int64_t c = mesh.branch_x[3 * b + 2];
        sum_x[at(l, r, c)] += branch_current[b];
        sum_x[at(l, r, c + 1)] += branch_current[b];
    }
    for (size_t b = 0; b < count_y; ++b) {
        const std::int64_t l = mesh.branch_y[3 * b];
        const std::int64_t r = mesh.branch_y[3 * b + 1];
        const std::int64_t c = mesh.branch_y[3 * b + 2];
        sum_y[at(l, r, c)] += branch_current[count_x + b];
        sum_y[at(l, r + 1, c)] += branch_current[count_x + b];
    }
    for (std::int64_t l = 0; l < mesh.layers; ++l) {
        const double thickness_mm = thickness_m[l] * 1e3;
        for (std::int64_t r = 0; r < mesh.rows; ++r) {
            const double hy = pitch_y[r] * 1e3;
            for (std::int64_t c = 0; c < mesh.cols; ++c) {
                const size_t cell = at(l, r, c);
                const std::int64_t node = mesh.node_of[cell];
                if (node < 0) {
                    continue;
                }
                const double hx = pitch_x[c] * 1e3;
                // NumPy divides a complex by a real as by a complex with zero
                // imaginary part: it multiplies by the reciprocal.  Doing the
                // same keeps the last bit.
                const double scale_x = 1.0 / (hy * thickness_mm);
                const double scale_y = 1.0 / (hx * thickness_mm);
                along_x[node] = (sum_x[cell] * 0.5) * scale_x;
                along_y[node] = (sum_y[cell] * 0.5) * scale_y;
            }
        }
    }
}

}  // namespace pcbcore::sheet
