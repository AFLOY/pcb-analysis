#include "pcbcore/sheet/rt0_moments.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "pcbcore/errors.hpp"

namespace pcbcore::sheet {

void polygon_potentials(const double* polygon, const int sides, const double x, const double y,
                        const double height, double& p0, double& p1x, double& p1y) {
    const double d = std::abs(height);
    const double d2 = d * d;
    double sum_tf = 0.0;
    double sum_beta = 0.0;
    p1x = 0.0;
    p1y = 0.0;
    for (int v = 0; v < sides; ++v) {
        const double ax = polygon[2 * v], ay = polygon[2 * v + 1];
        const int w = (v + 1) % sides;
        const double bx = polygon[2 * w], by = polygon[2 * w + 1];
        const double ex = bx - ax, ey = by - ay;
        const double length = std::sqrt(ex * ex + ey * ey);
        if (length <= 0.0) {
            continue;
        }
        const double tx = ex / length, ty = ey / length;
        const double nx = ty, ny = -tx;  // outward for counter-clockwise
        const double sx = ax - x, sy = ay - y;
        const double t0 = sx * nx + sy * ny;
        const double lm = sx * tx + sy * ty;
        const double lp = (bx - x) * tx + (by - y) * ty;
        const double r0sq = t0 * t0 + d2;
        const double rp = std::sqrt(r0sq + lp * lp);
        const double rm = std::sqrt(r0sq + lm * lm);
        const bool tiny = r0sq <= 1e-30 * std::max(lp * lp + lm * lm, 1e-300);
        double f = 0.0;
        if (!tiny) {
            const double gp = lp >= 0.0 ? rp + lp : r0sq / (rp - lp);
            const double gm = lm >= 0.0 ? rm + lm : r0sq / (rm - lm);
            f = std::log(gp / gm);
        }
        sum_tf += t0 * f;
        if (d > 0.0) {
            sum_beta += std::atan2(t0 * lp, r0sq + d * rp) - std::atan2(t0 * lm, r0sq + d * rm);
        }
        const double weight = r0sq * f + lp * rp - lm * rm;
        p1x += 0.5 * weight * nx;
        p1y += 0.5 * weight * ny;
    }
    p0 = sum_tf - d * sum_beta;
}

std::vector<double> rt0_near_moments(const double* outer_points, const double* outer_weights,
                                     const double* outer_centroid, const std::int64_t outer_count,
                                     const int quadrature, const double* inner_polygons,
                                     const std::int32_t* inner_sides, const double* inner_centroid,
                                     const std::int64_t inner_count, const std::int64_t* pair_outer,
                                     const std::int64_t* pair_inner, const std::int64_t pairs,
                                     const double* heights, const double* height_weights, const int height_count,
                                     const int threads) {
    for (std::int64_t p = 0; p < pairs; ++p) {
        if (pair_outer[p] < 0 || pair_outer[p] >= outer_count || pair_inner[p] < 0 || pair_inner[p] >= inner_count) {
            throw InvalidInput("a pair index is out of range");
        }
    }
    std::vector<double> out(static_cast<std::size_t>(pairs) * 9U, 0.0);
#pragma omp parallel for schedule(dynamic, 256) num_threads(std::max(threads, 1)) if (threads > 1)
    for (std::int64_t p = 0; p < pairs; ++p) {
        const std::int64_t e = pair_outer[p];
        const std::int64_t f = pair_inner[p];
        const double* poly = inner_polygons + 8 * f;
        const int sides = inner_sides[f];
        const double cex = outer_centroid[2 * e], cey = outer_centroid[2 * e + 1];
        const double cfx = inner_centroid[2 * f], cfy = inner_centroid[2 * f + 1];
        double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < quadrature; ++k) {
            const double wq = outer_weights[e * quadrature + k];
            if (wq == 0.0) {
                continue;
            }
            const double x = outer_points[2 * (e * quadrature + k)];
            const double y = outer_points[2 * (e * quadrature + k) + 1];
            double i0 = 0.0, i1 = 0.0, i2 = 0.0;
            for (int h = 0; h < height_count; ++h) {
                double p0, p1x, p1y;
                polygon_potentials(poly, sides, x, y, heights[h], p0, p1x, p1y);
                const double hw = height_weights[h];
                i0 += hw * p0;
                i1 += hw * (p1x + (x - cfx) * p0);
                i2 += hw * (p1y + (y - cfy) * p0);
            }
            const double o[3] = {wq, wq * (x - cex), wq * (y - cey)};
            const double in[3] = {i0, i1, i2};
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    acc[3 * a + b] += o[a] * in[b];
                }
            }
        }
        std::copy(acc, acc + 9, out.begin() + 9 * p);
    }
    return out;
}

}  // namespace pcbcore::sheet

namespace pcbcore::sheet {

std::vector<double> rt0_grid_moments(const std::int64_t* node_index_a, const double* node_moments_a,
                                     const std::int64_t count_a, const std::int64_t* node_index_b,
                                     const double* node_moments_b, const std::int64_t count_b,
                                     const int nodes_per_element, const double origin_x, const double origin_y,
                                     const double pitch, const std::int64_t columns, const double gap,
                                     const std::int64_t* pair_a, const std::int64_t* pair_b, const std::int64_t pairs,
                                     const int threads) {
    for (std::int64_t p = 0; p < pairs; ++p) {
        if (pair_a[p] < 0 || pair_a[p] >= count_a || pair_b[p] < 0 || pair_b[p] >= count_b) {
            throw InvalidInput("a pair index is out of range");
        }
    }
    const int k = nodes_per_element;
    const double g2 = gap * gap;
    std::vector<double> out(static_cast<std::size_t>(pairs) * 9U, 0.0);
#pragma omp parallel for schedule(dynamic, 256) num_threads(std::max(threads, 1)) if (threads > 1)
    for (std::int64_t p = 0; p < pairs; ++p) {
        const std::int64_t e = pair_a[p];
        const std::int64_t f = pair_b[p];
        double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < k; ++i) {
            const std::int64_t ni = node_index_a[e * k + i];
            if (ni < 0) {
                continue;
            }
            const double xi = origin_x + pitch * static_cast<double>(ni / columns);
            const double yi = origin_y + pitch * static_cast<double>(ni % columns);
            const double* mi = node_moments_a + 3 * (e * k + i);
            double t[3] = {0.0, 0.0, 0.0};  // sum_j G_ij m_j
            for (int j = 0; j < k; ++j) {
                const std::int64_t nj = node_index_b[f * k + j];
                if (nj < 0) {
                    continue;
                }
                const double dx = xi - (origin_x + pitch * static_cast<double>(nj / columns));
                const double dy = yi - (origin_y + pitch * static_cast<double>(nj % columns));
                const double r2 = dx * dx + dy * dy + g2;
                if (r2 <= 0.0) {
                    continue;
                }
                const double g = 1.0 / std::sqrt(r2);
                const double* mj = node_moments_b + 3 * (f * k + j);
                t[0] += g * mj[0];
                t[1] += g * mj[1];
                t[2] += g * mj[2];
            }
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    acc[3 * a + b] += mi[a] * t[b];
                }
            }
        }
        std::copy(acc, acc + 9, out.begin() + 9 * p);
    }
    return out;
}

}  // namespace pcbcore::sheet

namespace pcbcore::sheet {

void rt0_near_pairs(const double* centroid_a, const double* diameter_a, const std::int64_t count_a,
                    const double* centroid_b, const double* diameter_b, const std::int64_t count_b,
                    const double factor, const double floor, const int threads, std::vector<std::int64_t>& pair_a,
                    std::vector<std::int64_t>& pair_b) {
    pair_a.clear();
    pair_b.clear();
    if (count_a <= 0 || count_b <= 0) {
        return;
    }
    double max_diameter = 0.0;
    for (std::int64_t i = 0; i < count_a; ++i) {
        max_diameter = std::max(max_diameter, diameter_a[i]);
    }
    for (std::int64_t i = 0; i < count_b; ++i) {
        max_diameter = std::max(max_diameter, diameter_b[i]);
    }
    const double cell = std::max(factor * max_diameter, floor);
    if (!(cell > 0.0) || !std::isfinite(cell)) {
        throw InvalidInput("the near reach must be positive and finite");
    }
    double x0 = centroid_b[0], y0 = centroid_b[1], x1 = x0, y1 = y0;
    for (std::int64_t i = 0; i < count_b; ++i) {
        x0 = std::min(x0, centroid_b[2 * i]);
        x1 = std::max(x1, centroid_b[2 * i]);
        y0 = std::min(y0, centroid_b[2 * i + 1]);
        y1 = std::max(y1, centroid_b[2 * i + 1]);
    }
    const std::int64_t nx = static_cast<std::int64_t>((x1 - x0) / cell) + 1;
    const std::int64_t ny = static_cast<std::int64_t>((y1 - y0) / cell) + 1;
    // b's elements bucketed by cell, in index order within a cell (CSR)
    std::vector<std::int64_t> start(static_cast<std::size_t>(nx * ny + 1), 0), member(static_cast<std::size_t>(count_b));
    std::vector<std::int64_t> cell_of(static_cast<std::size_t>(count_b));
    for (std::int64_t i = 0; i < count_b; ++i) {
        const std::int64_t cx = std::min(nx - 1, static_cast<std::int64_t>((centroid_b[2 * i] - x0) / cell));
        const std::int64_t cy = std::min(ny - 1, static_cast<std::int64_t>((centroid_b[2 * i + 1] - y0) / cell));
        cell_of[i] = cx * ny + cy;
        ++start[cell_of[i] + 1];
    }
    for (std::int64_t c = 0; c < nx * ny; ++c) {
        start[c + 1] += start[c];
    }
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int64_t i = 0; i < count_b; ++i) {
            member[fill[cell_of[i]]++] = i;
        }
    }
    // each a's partners, found in parallel, then concatenated in a's order
    std::vector<std::vector<std::int64_t>> found(static_cast<std::size_t>(count_a));
#pragma omp parallel for schedule(dynamic, 256) num_threads(std::max(threads, 1)) if (threads > 1)
    for (std::int64_t e = 0; e < count_a; ++e) {
        const double ax = centroid_a[2 * e], ay = centroid_a[2 * e + 1];
        const std::int64_t cx = static_cast<std::int64_t>(std::floor((ax - x0) / cell));
        const std::int64_t cy = static_cast<std::int64_t>(std::floor((ay - y0) / cell));
        std::vector<std::int64_t>& mine = found[static_cast<std::size_t>(e)];
        for (std::int64_t i = std::max<std::int64_t>(cx - 1, 0); i <= std::min(cx + 1, nx - 1); ++i) {
            for (std::int64_t j = std::max<std::int64_t>(cy - 1, 0); j <= std::min(cy + 1, ny - 1); ++j) {
                const std::int64_t c = i * ny + j;
                for (std::int64_t s = start[c]; s < start[c + 1]; ++s) {
                    const std::int64_t f = member[s];
                    const double reach = std::max(factor * std::max(diameter_a[e], diameter_b[f]), floor);
                    const double dx = ax - centroid_b[2 * f], dy = ay - centroid_b[2 * f + 1];
                    if (dx * dx + dy * dy < reach * reach) {
                        mine.push_back(f);
                    }
                }
            }
        }
        std::sort(mine.begin(), mine.end());
    }
    std::size_t total = 0;
    for (const auto& v : found) {
        total += v.size();
    }
    pair_a.reserve(total);
    pair_b.reserve(total);
    for (std::int64_t e = 0; e < count_a; ++e) {
        for (const std::int64_t f : found[static_cast<std::size_t>(e)]) {
            pair_a.push_back(e);
            pair_b.push_back(f);
        }
    }
}

std::vector<double> rt0_point_moments(const double* points_a, const double* weights_a, const double* centroid_a,
                                      const std::int64_t count_a, const int quadrature_a, const double* points_b,
                                      const double* weights_b, const double* centroid_b, const std::int64_t count_b,
                                      const int quadrature_b, const std::int64_t* pair_a, const std::int64_t* pair_b,
                                      const std::int64_t pairs, const double* heights, const double* height_weights,
                                      const int height_count, const int threads) {
    for (std::int64_t p = 0; p < pairs; ++p) {
        if (pair_a[p] < 0 || pair_a[p] >= count_a || pair_b[p] < 0 || pair_b[p] >= count_b) {
            throw InvalidInput("a pair index is out of range");
        }
    }
    const int qa = quadrature_a, qb = quadrature_b;
    std::vector<double> out(static_cast<std::size_t>(pairs) * 9U, 0.0);
#pragma omp parallel for schedule(dynamic, 256) num_threads(std::max(threads, 1)) if (threads > 1)
    for (std::int64_t p = 0; p < pairs; ++p) {
        const std::int64_t e = pair_a[p];
        const std::int64_t f = pair_b[p];
        double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const double cax = centroid_a[2 * e], cay = centroid_a[2 * e + 1];
        const double cbx = centroid_b[2 * f], cby = centroid_b[2 * f + 1];
        for (int i = 0; i < qa; ++i) {
            const double xi = points_a[2 * (e * qa + i)], yi = points_a[2 * (e * qa + i) + 1];
            const double wi = weights_a[e * qa + i];
            double t[3] = {0.0, 0.0, 0.0};  // sum_j g_ij fb_j
            for (int j = 0; j < qb; ++j) {
                const double xj = points_b[2 * (f * qb + j)], yj = points_b[2 * (f * qb + j) + 1];
                const double dx = xi - xj, dy = yi - yj;
                const double r2 = dx * dx + dy * dy;
                double g = 0.0;
                for (int k = 0; k < height_count; ++k) {
                    g += height_weights[k] / std::sqrt(r2 + heights[k] * heights[k] + 1e-300);
                }
                g *= weights_b[f * qb + j];
                t[0] += g;
                t[1] += g * (xj - cbx);
                t[2] += g * (yj - cby);
            }
            const double fa[3] = {wi, wi * (xi - cax), wi * (yi - cay)};
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    acc[3 * a + b] += fa[a] * t[b];
                }
            }
        }
        std::copy(acc, acc + 9, out.begin() + 9 * p);
    }
    return out;
}

}  // namespace pcbcore::sheet
