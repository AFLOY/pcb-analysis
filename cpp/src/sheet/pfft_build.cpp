#include "pcbcore/sheet/pfft_build.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <utility>

#include "pcbcore/errors.hpp"
#include "pcbcore/sheet/hoer_love.hpp"

namespace pcbcore::sheet {

namespace {

using std::size_t;

constexpr double kPi = 3.141592653589793;
const double kMu0Over4Pi = (4.0e-7 * kPi) / (4.0 * kPi);

int team_size(const int threads, const std::int64_t tasks) {
    return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(threads, tasks)));
}

// fftfreq(n, d=1/n): 0, 1, ..., then the negative offsets.
std::int64_t wrapped_offset(const std::int64_t index, const std::int64_t count) {
    const std::int64_t positive = (count - 1) / 2 + 1;
    return index < positive ? index : index - count;
}

// NumPy's float modulo: fmod, moved into the divisor's sign.
double numpy_mod(const double a, const double b) {
    double mod = std::fmod(a, b);
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) {
            mod += b;
        }
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

// Grid indices and Lagrange weights interpolating at ``coordinate``.
void lagrange_stencil(const double coordinate, const double origin, const double pitch, const std::int64_t count,
                      const int order, std::int64_t* const nodes, double* const weights) {
    const double x = (coordinate - origin) / pitch;
    std::int64_t first = static_cast<std::int64_t>(std::floor(x)) - (order - 1) / 2;
    first = std::clamp<std::int64_t>(first, 0, count - 1 - order);
    for (int j = 0; j <= order; ++j) {
        nodes[j] = first + j;
        weights[j] = 1.0;
    }
    for (int j = 0; j <= order; ++j) {
        for (int k = 0; k <= order; ++k) {
            if (k != j) {
                weights[j] *= (x - static_cast<double>(nodes[k])) / static_cast<double>(nodes[j] - nodes[k]);
            }
        }
    }
}

// The projection of each bar (centre, extents) onto the grid, sampled at
// Gauss points over its extent: three along the longer side, two across.
CsrMatrix projection_matrix(const ProjectionGridSpec& grid, const bool along_x, const std::int64_t n,
                            const double* const cx, const double* const cy, const double* const ex,
                            const double* const ey) {
    const double g3 = std::sqrt(3.0 / 5.0);
    const double points3[3] = {-g3 / 2.0, 0.0 / 2.0, g3 / 2.0};
    const double weights3[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
    const double points2[2] = {-1.0 / (2.0 * std::sqrt(3.0)), 1.0 / (2.0 * std::sqrt(3.0))};
    const double weights2[2] = {0.5, 0.5};
    const double* px = along_x ? points3 : points2;
    const double* wxq = along_x ? weights3 : weights2;
    const int nx_points = along_x ? 3 : 2;
    const double* py = along_x ? points2 : points3;
    const double* wyq = along_x ? weights2 : weights3;
    const int ny_points = along_x ? 2 : 3;
    const int order = grid.order;
    const int width = order + 1;

    CsrMatrix m;
    m.rows = n;
    m.cols = grid.nodes_x * grid.nodes_y;
    m.indptr.assign(static_cast<size_t>(n) + 1U, 0);
    std::vector<std::int64_t> ix(static_cast<size_t>(width));
    std::vector<std::int64_t> iy(static_cast<size_t>(width));
    std::vector<double> wx(static_cast<size_t>(width));
    std::vector<double> wy(static_cast<size_t>(width));
    std::vector<std::pair<std::int64_t, double>> row;
    for (std::int64_t b = 0; b < n; ++b) {
        row.clear();
        for (int a = 0; a < nx_points; ++a) {
            for (int c = 0; c < ny_points; ++c) {
                lagrange_stencil(cx[b] + px[a] * ex[b], grid.origin_x, grid.pitch, grid.nodes_x, order, ix.data(),
                                 wx.data());
                lagrange_stencil(cy[b] + py[c] * ey[b], grid.origin_y, grid.pitch, grid.nodes_y, order, iy.data(),
                                 wy.data());
                const double q = wxq[a] * wyq[c];
                for (int s = 0; s < width; ++s) {
                    for (int t = 0; t < width; ++t) {
                        const std::int64_t node = iy[static_cast<size_t>(s)] * grid.nodes_x + ix[static_cast<size_t>(t)];
                        const double value = q * (wy[static_cast<size_t>(s)] * wx[static_cast<size_t>(t)]);
                        // Blocks add up in block order, as the sparse sum of the blocks does.
                        auto found = std::find_if(row.begin(), row.end(),
                                                  [node](const auto& entry) { return entry.first == node; });
                        if (found == row.end()) {
                            row.emplace_back(node, value);
                        } else {
                            found->second += value;
                        }
                    }
                }
            }
        }
        std::sort(row.begin(), row.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        for (const auto& [node, value] : row) {
            m.indices.push_back(node);
            m.data.push_back(value);
        }
        m.indptr[static_cast<size_t>(b) + 1U] = static_cast<std::int64_t>(m.data.size());
    }
    return m;
}

// Index pairs of a sorted 1-D coordinate array within the near radius plus
// both half extents.
void near_pairs(const std::vector<double>& coords, const std::vector<double>& half, const double radius,
                std::vector<std::int64_t>& out_i, std::vector<std::int64_t>& out_j) {
    out_i.clear();
    out_j.clear();
    if (coords.empty()) {
        return;
    }
    const double widest = *std::max_element(half.begin(), half.end());
    const auto n = static_cast<std::int64_t>(coords.size());
    for (std::int64_t i = 0; i < n; ++i) {
        const double reach = radius + half[static_cast<size_t>(i)] + widest;
        const auto lo = std::lower_bound(coords.begin(), coords.end(), coords[static_cast<size_t>(i)] - reach);
        const auto hi = std::upper_bound(coords.begin(), coords.end(), coords[static_cast<size_t>(i)] + reach);
        for (auto it = lo; it != hi; ++it) {
            const auto j = static_cast<std::int64_t>(it - coords.begin());
            if (std::abs(coords[static_cast<size_t>(j)] - coords[static_cast<size_t>(i)]) <=
                radius + half[static_cast<size_t>(i)] + half[static_cast<size_t>(j)]) {
                out_i.push_back(i);
                out_j.push_back(j);
            }
        }
    }
}

// One representative per distinct quantised row (the first in list order)
// and each row's representative.  An open-addressing hash of the packed key
// rows, walked in list order: the first row seen with a key represents it,
// which is the representative a stable sort of the keys picks.
void unique_rows(const std::vector<std::vector<std::int64_t>>& keys, std::vector<std::int64_t>& first,
                 std::vector<std::int64_t>& inverse) {
    const size_t n = keys.empty() ? 0U : keys[0].size();
    const size_t width = keys.size();
    std::vector<std::int64_t> packed(n * width);
    for (size_t k = 0; k < n; ++k) {
        for (size_t c = 0; c < width; ++c) {
            packed[k * width + c] = keys[c][k];
        }
    }
    auto row = [&packed, width](const size_t k) { return packed.data() + k * width; };
    auto hash = [&row, width](const size_t k) {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        const std::int64_t* x = row(k);
        for (size_t c = 0; c < width; ++c) {
            std::uint64_t z = h ^ static_cast<std::uint64_t>(x[c]);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            h = z ^ (z >> 31);
        }
        return h;
    };
    size_t capacity = 16;
    while (capacity < 2 * n) {
        capacity <<= 1U;
    }
    std::vector<std::int64_t> slot(capacity, -1);  // list position of the first row with that key
    std::vector<std::int64_t> group_of_slot(capacity, -1);
    first.clear();
    inverse.assign(n, 0);
    for (size_t k = 0; k < n; ++k) {
        size_t s = static_cast<size_t>(hash(k)) & (capacity - 1);
        while (true) {
            const std::int64_t occupant = slot[s];
            if (occupant < 0) {
                slot[s] = static_cast<std::int64_t>(k);
                group_of_slot[s] = static_cast<std::int64_t>(first.size());
                first.push_back(static_cast<std::int64_t>(k));
                break;
            }
            if (std::equal(row(k), row(k) + width, row(static_cast<size_t>(occupant)))) {
                break;
            }
            s = (s + 1) & (capacity - 1);
        }
        inverse[k] = group_of_slot[s];
    }
}

std::vector<std::int64_t> quantised(const std::vector<double>& values) {
    std::vector<std::int64_t> out(values.size());
    for (size_t k = 0; k < values.size(); ++k) {
        out[k] = static_cast<std::int64_t>(std::nearbyint(values[k] / 1.0e-12));
    }
    return out;
}

struct PairStructure {
    std::vector<std::int64_t> bi;
    std::vector<std::int64_t> bj;
    std::vector<std::int64_t> exact_first;
    std::vector<std::int64_t> exact_inverse;
    std::vector<std::int64_t> grid_first;
    std::vector<std::int64_t> grid_inverse;
};

PairStructure pair_structure(const ProjectionGridSpec& grid, const std::int64_t n_rows, const std::int64_t n_cols,
                             const double* const cx, const double* const cy, const double* const half_x,
                             const double* const half_y) {
    const double radius = static_cast<double>(grid.near_radius_cells) * grid.pitch * (1.0 + 1.0e-9);
    std::vector<double> column_coords(cx, cx + n_cols);
    std::vector<double> column_half(half_x, half_x + n_cols);
    std::vector<double> row_coords(static_cast<size_t>(n_rows));
    std::vector<double> row_half(static_cast<size_t>(n_rows));
    for (std::int64_t r = 0; r < n_rows; ++r) {
        row_coords[static_cast<size_t>(r)] = cy[r * n_cols];
        row_half[static_cast<size_t>(r)] = half_y[r * n_cols];
    }
    std::vector<std::int64_t> ci;
    std::vector<std::int64_t> cj;
    std::vector<std::int64_t> ri;
    std::vector<std::int64_t> rj;
    near_pairs(column_coords, column_half, radius, ci, cj);
    near_pairs(row_coords, row_half, radius, ri, rj);
    PairStructure p;
    p.bi.reserve(ri.size() * ci.size());
    p.bj.reserve(ri.size() * ci.size());
    for (size_t r = 0; r < ri.size(); ++r) {
        for (size_t c = 0; c < ci.size(); ++c) {
            p.bi.push_back(ri[r] * n_cols + ci[c]);
            p.bj.push_back(rj[r] * n_cols + cj[c]);
        }
    }
    const size_t count = p.bi.size();
    std::vector<double> hxi(count), hyi(count), hxj(count), hyj(count), dx(count), dy(count), fxi(count), fyi(count);
    for (size_t k = 0; k < count; ++k) {
        const auto i = static_cast<size_t>(p.bi[k]);
        const auto j = static_cast<size_t>(p.bj[k]);
        hxi[k] = half_x[i];
        hyi[k] = half_y[i];
        hxj[k] = half_x[j];
        hyj[k] = half_y[j];
        dx[k] = cx[j] - cx[i];
        dy[k] = cy[j] - cy[i];
        fxi[k] = numpy_mod(cx[i] - grid.origin_x, grid.pitch);
        fyi[k] = numpy_mod(cy[i] - grid.origin_y, grid.pitch);
    }
    // The exact term depends on the two bars and their offset; the grid term
    // also on where the first bar sits relative to the grid lines.
    unique_rows({quantised(hxi), quantised(hyi), quantised(hxj), quantised(hyj), quantised(dx), quantised(dy)},
                p.exact_first, p.exact_inverse);
    unique_rows({quantised(fxi), quantised(fyi), quantised(hxi), quantised(hyi), quantised(hxj), quantised(hyj),
                 quantised(dx), quantised(dy)},
                p.grid_first, p.grid_inverse);
    return p;
}

// P_i K P_j^T for each listed pair: what the FFT path credits it with.
std::vector<double> grid_pair_coupling(const ProjectionGridSpec& grid, const CsrMatrix& projection,
                                       const std::vector<std::int64_t>& pair_i,
                                       const std::vector<std::int64_t>& pair_j, const double separation,
                                       const int threads) {
    const std::vector<double> table = pfft_kernel_table(grid, separation);
    const std::int64_t rows = 2 * grid.nodes_y;
    const std::int64_t cols = 2 * grid.nodes_x;
    const auto count = static_cast<std::int64_t>(pair_i.size());
    std::vector<double> out(static_cast<size_t>(count));
    const int team = team_size(threads, count / 1024 + 1);
#pragma omp parallel for schedule(static) num_threads(team) if (team > 1)
    for (std::int64_t p = 0; p < count; ++p) {
        const auto i = static_cast<size_t>(pair_i[static_cast<size_t>(p)]);
        const auto j = static_cast<size_t>(pair_j[static_cast<size_t>(p)]);
        double acc = 0.0;
        for (std::int64_t a = projection.indptr[i]; a < projection.indptr[i + 1]; ++a) {
            const std::int64_t node_a = projection.indices[static_cast<size_t>(a)];
            const std::int64_t ya = node_a / grid.nodes_x;
            const std::int64_t xa = node_a - ya * grid.nodes_x;
            double inner = 0.0;
            for (std::int64_t b = projection.indptr[j]; b < projection.indptr[j + 1]; ++b) {
                const std::int64_t node_b = projection.indices[static_cast<size_t>(b)];
                const std::int64_t yb = node_b / grid.nodes_x;
                const std::int64_t xb = node_b - yb * grid.nodes_x;
                std::int64_t dy = (yb - ya) % rows;
                if (dy < 0) {
                    dy += rows;
                }
                std::int64_t dx = (xb - xa) % cols;
                if (dx < 0) {
                    dx += cols;
                }
                inner += projection.data[static_cast<size_t>(b)] * table[static_cast<size_t>(dy * cols + dx)];
            }
            acc += projection.data[static_cast<size_t>(a)] * inner;
        }
        out[static_cast<size_t>(p)] = acc;
    }
    return out;
}

// The square (planes count) CSR of per-plane-pair blocks, row by row: for
// each row, block columns in order, each block's entries in pair order.
CsrMatrix assemble_blocks(const std::int64_t planes, const std::int64_t count, const PairStructure& pairs,
                          const std::vector<std::vector<double>>& block_values, const std::vector<bool>* keep) {
    std::vector<std::vector<std::int64_t>> by_row(static_cast<size_t>(count));
    for (size_t k = 0; k < pairs.bi.size(); ++k) {
        if (keep == nullptr || (*keep)[k]) {
            by_row[static_cast<size_t>(pairs.bi[k])].push_back(static_cast<std::int64_t>(k));
        }
    }
    CsrMatrix m;
    m.rows = planes * count;
    m.cols = planes * count;
    m.indptr.assign(static_cast<size_t>(m.rows) + 1U, 0);
    for (std::int64_t a = 0; a < planes; ++a) {
        for (std::int64_t r = 0; r < count; ++r) {
            for (std::int64_t b = 0; b < planes; ++b) {
                const auto& values = block_values[static_cast<size_t>(a * planes + b)];
                for (const std::int64_t k : by_row[static_cast<size_t>(r)]) {
                    m.indices.push_back(b * count + pairs.bj[static_cast<size_t>(k)]);
                    m.data.push_back(values[static_cast<size_t>(k)]);
                }
            }
            m.indptr[static_cast<size_t>(a * count + r) + 1U] = static_cast<std::int64_t>(m.data.size());
        }
    }
    return m;
}

void check_grid(const ProjectionGridSpec& grid) {
    if (grid.nodes_x < grid.order + 1 || grid.nodes_y < grid.order + 1 || !(grid.pitch > 0.0) || grid.order < 1) {
        throw InvalidInput("the projection grid is too small for the stencil");
    }
}

}  // namespace

std::vector<double> pfft_kernel_table(const ProjectionGridSpec& grid, const double separation) {
    const std::int64_t rows = 2 * grid.nodes_y;
    const std::int64_t cols = 2 * grid.nodes_x;
    std::vector<double> table(static_cast<size_t>(rows * cols));
    for (std::int64_t r = 0; r < rows; ++r) {
        const double dy = static_cast<double>(wrapped_offset(r, rows)) * grid.pitch;
        for (std::int64_t c = 0; c < cols; ++c) {
            const double dx = static_cast<double>(wrapped_offset(c, cols)) * grid.pitch;
            const double radius = std::sqrt(dx * dx + dy * dy + separation * separation);
            table[static_cast<size_t>(r * cols + c)] = radius > 0.0 ? kMu0Over4Pi / radius : 0.0;
        }
    }
    return table;
}

PfftFamilyParts pfft_inplane_family(const ProjectionGridSpec& grid, const bool axis_x, const bool sample_along_x,
                                    const std::int64_t n_rows, const std::int64_t n_cols, const double* const cx,
                                    const double* const cy, const double* const length, const double* const width,
                                    const std::int64_t layers, const double* const layer_z,
                                    const double* const layer_thickness, const double* const separation,
                                    const int threads) {
    check_grid(grid);
    const std::int64_t count = n_rows * n_cols;
    std::vector<double> ex(static_cast<size_t>(count));
    std::vector<double> ey(static_cast<size_t>(count));
    std::vector<double> half_x(static_cast<size_t>(count));
    std::vector<double> half_y(static_cast<size_t>(count));
    for (std::int64_t k = 0; k < count; ++k) {
        ex[static_cast<size_t>(k)] = axis_x ? length[k] : width[k];
        ey[static_cast<size_t>(k)] = axis_x ? width[k] : length[k];
        half_x[static_cast<size_t>(k)] = ex[static_cast<size_t>(k)] / 2.0;
        half_y[static_cast<size_t>(k)] = ey[static_cast<size_t>(k)] / 2.0;
    }
    PfftFamilyParts parts;
    parts.projection = projection_matrix(grid, sample_along_x, count, cx, cy, ex.data(), ey.data());
    parts.self_value.assign(static_cast<size_t>(layers * count), 0.0);
    if (count == 0) {
        parts.correction = assemble_blocks(layers, 0, PairStructure{}, {}, nullptr);
        parts.near_exact = parts.correction;
        return parts;
    }
    const PairStructure pairs = pair_structure(grid, n_rows, n_cols, cx, cy, half_x.data(), half_y.data());
    const size_t total = pairs.bi.size();
    const double reach = static_cast<double>(grid.preconditioner_radius_cells) * grid.pitch * (1.0 + 1.0e-9);
    std::vector<bool> near_mask(total);
    for (size_t k = 0; k < total; ++k) {
        const auto i = static_cast<size_t>(pairs.bi[k]);
        const auto j = static_cast<size_t>(pairs.bj[k]);
        near_mask[k] = std::abs(cx[j] - cx[i]) <= reach + half_x[i] + half_x[j] &&
                       std::abs(cy[j] - cy[i]) <= reach + half_y[i] + half_y[j];
    }
    const size_t unique_exact = pairs.exact_first.size();
    std::vector<std::int64_t> gi;
    std::vector<std::int64_t> gj;
    for (const std::int64_t f : pairs.grid_first) {
        gi.push_back(pairs.bi[static_cast<size_t>(f)]);
        gj.push_back(pairs.bj[static_cast<size_t>(f)]);
    }
    std::map<std::tuple<double, double, double>, std::vector<double>> exact_cache;
    std::map<double, std::vector<double>> grid_cache;
    std::vector<std::vector<double>> correction_blocks;
    std::vector<std::vector<double>> near_blocks;
    for (std::int64_t a = 0; a < layers; ++a) {
        for (std::int64_t b = 0; b < layers; ++b) {
            const double sep = separation[a * layers + b];
            const double dz = layer_z[b] - layer_z[a];
            const double thin = std::min(layer_thickness[a], layer_thickness[b]);
            const double thick = std::max(layer_thickness[a], layer_thickness[b]);
            const auto key = std::make_tuple(sep, thin, thick);
            if (exact_cache.find(key) == exact_cache.end()) {
                std::vector<double> la(unique_exact), wa(unique_exact), lb(unique_exact), wb(unique_exact);
                std::vector<double> du(unique_exact), dv(unique_exact);
                for (size_t u = 0; u < unique_exact; ++u) {
                    const auto f = static_cast<size_t>(pairs.exact_first[u]);
                    const auto i = static_cast<size_t>(pairs.bi[f]);
                    const auto j = static_cast<size_t>(pairs.bj[f]);
                    la[u] = length[i];
                    wa[u] = width[i];
                    lb[u] = length[j];
                    wb[u] = width[j];
                    du[u] = axis_x ? cx[j] - cx[i] : cy[j] - cy[i];
                    dv[u] = axis_x ? cy[j] - cy[i] : cx[j] - cx[i];
                }
                const double tz[1] = {thin};
                const double tk[1] = {thick};
                const double ad[1] = {std::abs(dz)};
                std::vector<double> values(unique_exact);
                closed_form_arrays(static_cast<std::int64_t>(unique_exact), BarArrays{la.data(), wa.data(), tz, 1, 1, 0},
                                   BarArrays{lb.data(), wb.data(), tk, 1, 1, 0},
                                   OffsetArrays{du.data(), dv.data(), ad, 1, 1, 0}, values.data(), threads);
                exact_cache.emplace(key, std::move(values));
            }
            if (grid_cache.find(sep) == grid_cache.end()) {
                std::vector<double> coupling = grid_pair_coupling(grid, parts.projection, gi, gj, sep, threads);
                for (size_t u = 0; u < coupling.size(); ++u) {
                    coupling[u] = length[static_cast<size_t>(gi[u])] * length[static_cast<size_t>(gj[u])] * coupling[u];
                }
                grid_cache.emplace(sep, std::move(coupling));
            }
            const std::vector<double>& exact_unique = exact_cache.at(key);
            const std::vector<double>& grid_unique = grid_cache.at(sep);
            std::vector<double> corr(total);
            std::vector<double> near(total);
            for (size_t k = 0; k < total; ++k) {
                const double exact = exact_unique[static_cast<size_t>(pairs.exact_inverse[k])];
                corr[k] = exact - grid_unique[static_cast<size_t>(pairs.grid_inverse[k])];
                near[k] = exact;
                if (a == b && pairs.bi[k] == pairs.bj[k]) {
                    parts.self_value[static_cast<size_t>(a * count + pairs.bi[k])] = exact;
                }
            }
            correction_blocks.push_back(std::move(corr));
            near_blocks.push_back(std::move(near));
        }
    }
    parts.correction = assemble_blocks(layers, count, pairs, correction_blocks, nullptr);
    parts.near_exact = assemble_blocks(layers, count, pairs, near_blocks, &near_mask);
    return parts;
}

PfftFamilyParts pfft_vertical_family(const ProjectionGridSpec& grid, const bool sample_along_x,
                                     const std::int64_t rows, const std::int64_t cols, const double* const centre_x,
                                     const double* const centre_y, const double* const pitch_x,
                                     const double* const pitch_y, const std::int64_t levels, const double* const span,
                                     const double* const z_mid, const double* const separation, const int threads) {
    check_grid(grid);
    const std::int64_t count = rows * cols;
    std::vector<double> hx(static_cast<size_t>(count));
    std::vector<double> hy(static_cast<size_t>(count));
    std::vector<double> half_x(static_cast<size_t>(count));
    std::vector<double> half_y(static_cast<size_t>(count));
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const auto k = static_cast<size_t>(r * cols + c);
            hx[k] = pitch_x[c];
            hy[k] = pitch_y[r];
            half_x[k] = hx[k] / 2.0;
            half_y[k] = hy[k] / 2.0;
        }
    }
    PfftFamilyParts parts;
    parts.projection = projection_matrix(grid, sample_along_x, count, centre_x, centre_y, hx.data(), hy.data());
    parts.self_value.assign(static_cast<size_t>(levels * count), 0.0);
    const PairStructure pairs = pair_structure(grid, rows, cols, centre_x, centre_y, half_x.data(), half_y.data());
    const size_t total = pairs.bi.size();
    const double reach = static_cast<double>(grid.preconditioner_radius_cells) * grid.pitch * (1.0 + 1.0e-9);
    std::vector<bool> near_mask(total);
    for (size_t k = 0; k < total; ++k) {
        const auto i = static_cast<size_t>(pairs.bi[k]);
        const auto j = static_cast<size_t>(pairs.bj[k]);
        near_mask[k] = std::abs(centre_x[j] - centre_x[i]) <= reach + (hx[i] + hx[j]) / 2.0 &&
                       std::abs(centre_y[j] - centre_y[i]) <= reach + (hy[i] + hy[j]) / 2.0;
    }
    const size_t unique_exact = pairs.exact_first.size();
    std::vector<std::int64_t> gi;
    std::vector<std::int64_t> gj;
    for (const std::int64_t f : pairs.grid_first) {
        gi.push_back(pairs.bi[static_cast<size_t>(f)]);
        gj.push_back(pairs.bj[static_cast<size_t>(f)]);
    }
    std::vector<double> wi(unique_exact), ti(unique_exact), wj(unique_exact), tj(unique_exact), dv(unique_exact),
        dw(unique_exact);
    for (size_t u = 0; u < unique_exact; ++u) {
        const auto f = static_cast<size_t>(pairs.exact_first[u]);
        const auto i = static_cast<size_t>(pairs.bi[f]);
        const auto j = static_cast<size_t>(pairs.bj[f]);
        wi[u] = hx[i];
        ti[u] = hy[i];
        wj[u] = hx[j];
        tj[u] = hy[j];
        dv[u] = centre_x[j] - centre_x[i];
        dw[u] = centre_y[j] - centre_y[i];
    }
    std::map<double, std::vector<double>> grid_cache;
    std::vector<std::vector<double>> correction_blocks;
    std::vector<std::vector<double>> near_blocks;
    for (std::int64_t a = 0; a < levels; ++a) {
        for (std::int64_t b = 0; b < levels; ++b) {
            const double sep = separation[a * levels + b];
            const double sa[1] = {span[a]};
            const double sb[1] = {span[b]};
            const double du[1] = {z_mid[b] - z_mid[a]};
            std::vector<double> exact_unique(unique_exact);
            closed_form_arrays(static_cast<std::int64_t>(unique_exact), BarArrays{sa, wi.data(), ti.data(), 0, 1, 1},
                               BarArrays{sb, wj.data(), tj.data(), 0, 1, 1}, OffsetArrays{du, dv.data(), dw.data(), 0, 1, 1},
                               exact_unique.data(), threads);
            if (grid_cache.find(sep) == grid_cache.end()) {
                grid_cache.emplace(sep, grid_pair_coupling(grid, parts.projection, gi, gj, sep, threads));
            }
            const std::vector<double>& grid_unique = grid_cache.at(sep);
            std::vector<double> corr(total);
            std::vector<double> near(total);
            for (size_t k = 0; k < total; ++k) {
                const double exact = exact_unique[static_cast<size_t>(pairs.exact_inverse[k])];
                const double approximate = span[a] * span[b] * grid_unique[static_cast<size_t>(pairs.grid_inverse[k])];
                corr[k] = exact - approximate;
                near[k] = exact;
                if (a == b && pairs.bi[k] == pairs.bj[k]) {
                    parts.self_value[static_cast<size_t>(a * count + pairs.bi[k])] = exact;
                }
            }
            correction_blocks.push_back(std::move(corr));
            near_blocks.push_back(std::move(near));
        }
    }
    parts.correction = assemble_blocks(levels, count, pairs, correction_blocks, nullptr);
    parts.near_exact = assemble_blocks(levels, count, pairs, near_blocks, &near_mask);
    return parts;
}

}  // namespace pcbcore::sheet
