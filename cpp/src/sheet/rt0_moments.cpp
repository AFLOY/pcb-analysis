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
