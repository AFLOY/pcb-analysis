// Point-in-solid classification by the generalized winding number.
//
// For a closed, outward-oriented triangle mesh the sum of the signed solid
// angles subtended by its triangles at a query point is 4*pi inside and 0
// outside (Van Oosterom & Strackee, 1983, for the solid angle of one
// triangle).  Unlike ray parity it has no degenerate ray/edge cases and it
// degrades gracefully on slightly open meshes: the value stays near 0 or 1
// away from the hole.  The query points of this repository lie on regular
// grids, so the work is embarrassingly parallel over points; the triangles
// of one solid (hundreds to a few thousand) are read from cache.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr double kFourPi = 4.0 * 3.14159265358979323846;

struct Triangle {
    double a[3], b[3], c[3];
    double lo[3], hi[3];
};

inline double solid_angle(const Triangle& t, const double* p) {
    // Vectors from the query point to the vertices.
    const double ax = t.a[0] - p[0], ay = t.a[1] - p[1], az = t.a[2] - p[2];
    const double bx = t.b[0] - p[0], by = t.b[1] - p[1], bz = t.b[2] - p[2];
    const double cx = t.c[0] - p[0], cy = t.c[1] - p[1], cz = t.c[2] - p[2];
    const double la = std::sqrt(ax * ax + ay * ay + az * az);
    const double lb = std::sqrt(bx * bx + by * by + bz * bz);
    const double lc = std::sqrt(cx * cx + cy * cy + cz * cz);
    // a . (b x c)
    const double numerator = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    const double denominator = la * lb * lc + (ax * bx + ay * by + az * bz) * lc +
                               (ax * cx + ay * cy + az * cz) * lb + (bx * cx + by * cy + bz * cz) * la;
    return 2.0 * std::atan2(numerator, denominator);
}

std::vector<Triangle> load_triangles(const ArrF64& triangles) {
    if (triangles.ndim() != 3 || triangles.shape(1) != 3 || triangles.shape(2) != 3) {
        throw std::invalid_argument("triangles must have shape (n, 3, 3)");
    }
    const py::ssize_t count = triangles.shape(0);
    const double* data = triangles.data();
    std::vector<Triangle> out(static_cast<std::size_t>(count));
    for (py::ssize_t i = 0; i < count; ++i) {
        Triangle& t = out[static_cast<std::size_t>(i)];
        const double* base = data + i * 9;
        for (int k = 0; k < 3; ++k) {
            t.a[k] = base[k];
            t.b[k] = base[3 + k];
            t.c[k] = base[6 + k];
            t.lo[k] = std::min({t.a[k], t.b[k], t.c[k]});
            t.hi[k] = std::max({t.a[k], t.b[k], t.c[k]});
        }
    }
    return out;
}

// Winding number (solid angle / 4 pi) of every point: ~1 inside, ~0 outside.
py::array_t<double> winding_numbers(const ArrF64& points, const ArrF64& triangles, int threads) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (n, 3)");
    }
    const std::vector<Triangle> mesh = load_triangles(triangles);
    const py::ssize_t count = points.shape(0);
    py::array_t<double> result(count);
    double* out = result.mutable_data();
    const double* pts = points.data();
    const int tri_count = static_cast<int>(mesh.size());
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
#endif
        for (py::ssize_t i = 0; i < count; ++i) {
            const double* p = pts + i * 3;
            double total = 0.0;
            for (int k = 0; k < tri_count; ++k) {
                total += solid_angle(mesh[static_cast<std::size_t>(k)], p);
            }
            out[i] = total / kFourPi;
        }
    }
    return result;
}

// Inside test with a bounding-box prefilter, the common call.
py::array_t<bool> contains(const ArrF64& points, const ArrF64& triangles, double threshold, int threads) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (n, 3)");
    }
    const std::vector<Triangle> mesh = load_triangles(triangles);
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (const Triangle& t : mesh) {
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], t.lo[k]);
            hi[k] = std::max(hi[k], t.hi[k]);
        }
    }
    const py::ssize_t count = points.shape(0);
    py::array_t<bool> result(count);
    bool* out = result.mutable_data();
    const double* pts = points.data();
    const int tri_count = static_cast<int>(mesh.size());
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        omp_set_dynamic(0);
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads > 0 ? threads : 1)
#endif
        for (py::ssize_t i = 0; i < count; ++i) {
            const double* p = pts + i * 3;
            bool inside_box = true;
            for (int k = 0; k < 3; ++k) {
                inside_box = inside_box && p[k] >= lo[k] && p[k] <= hi[k];
            }
            if (!inside_box) {
                out[i] = false;
                continue;
            }
            double total = 0.0;
            for (int k = 0; k < tri_count; ++k) {
                total += solid_angle(mesh[static_cast<std::size_t>(k)], p);
            }
            out[i] = total / kFourPi >= threshold;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Planar section of the mesh rasterised to exact per-cell area coverage.
//
// A copper layer of a 2.5D board is the section of the copper solids at the
// layer's centre z.  Every triangle that crosses the plane contributes one
// oriented segment (the section of an outward-oriented closed surface is a
// set of closed loops, counter-clockwise around copper, so no stitching is
// needed).  The segments are accumulated onto the cell grid with the
// signed-area scheme of font rasterisers (Raph Levien's font-rs): each
// segment adds its area and cover terms to an accumulation buffer, a prefix
// sum along x turns that into the exact fraction of each cell inside the
// loops.  Cost is linear in the crossing triangles and the cells they touch,
// independent of the solid's bounding box.

struct Segment {
    double x0, y0, x1, y1;  // in cell units, y up
};

inline void accumulate_line(std::vector<double>& acc, int rows, int cols, double px0, double py0, double px1,
                            double py1) {
    // Clamp x into [0, cols]: a segment left of the grid still winds every
    // cell to its right, which column 0 then carries; rows outside are dropped.
    const double width = static_cast<double>(cols);
    px0 = std::min(std::max(px0, 0.0), width);
    px1 = std::min(std::max(px1, 0.0), width);
    if (std::fabs(py0 - py1) <= 1.0e-12) return;
    double dir = 1.0;
    if (py0 > py1) {
        std::swap(px0, px1);
        std::swap(py0, py1);
        dir = -1.0;
    }
    const double dxdy = (px1 - px0) / (py1 - py0);
    double x = px0;
    int y_start = static_cast<int>(std::floor(py0));
    if (py0 < 0.0) {
        x -= py0 * dxdy;
        y_start = 0;
    }
    const int y_end = std::min(rows, static_cast<int>(std::ceil(py1)));
    const int stride = cols + 2;
    for (int y = y_start; y < y_end; ++y) {
        const double dy = std::min(static_cast<double>(y + 1), py1) - std::max(static_cast<double>(y), py0);
        const double xnext = x + dxdy * dy;
        const double d = dy * dir;
        double xa = x, xb = xnext;
        if (xa > xb) std::swap(xa, xb);
        const double xa_floor = std::floor(xa);
        const int xa_i = static_cast<int>(xa_floor);
        const double xb_ceil = std::ceil(xb);
        const int xb_i = static_cast<int>(xb_ceil);
        double* line = acc.data() + static_cast<std::size_t>(y) * stride;
        if (xb_i <= xa_i + 1) {
            const double xmf = 0.5 * (x + xnext) - xa_floor;
            line[xa_i] += d - d * xmf;
            line[xa_i + 1] += d * xmf;
        } else {
            const double s = 1.0 / (xb - xa);
            const double xa_f = xa - xa_floor;
            const double a0 = 0.5 * s * (1.0 - xa_f) * (1.0 - xa_f);
            const double xb_f = xb - xb_ceil + 1.0;
            const double am = 0.5 * s * xb_f * xb_f;
            line[xa_i] += d * a0;
            if (xb_i == xa_i + 2) {
                line[xa_i + 1] += d * (1.0 - a0 - am);
            } else {
                const double a1 = s * (1.5 - xa_f);
                line[xa_i + 1] += d * (a1 - a0);
                for (int xi = xa_i + 2; xi < xb_i - 1; ++xi) line[xi] += d * s;
                const double a2 = a1 + static_cast<double>(xb_i - xa_i - 3) * s;
                line[xb_i - 1] += d * (1.0 - a2 - am);
            }
            line[xb_i] += d * am;
        }
        x = xnext;
    }
}

// Oriented section segments of the triangles at height z, in metres.
std::vector<Segment> plane_section(const std::vector<Triangle>& mesh, double z) {
    std::vector<Segment> segments;
    for (const Triangle& t : mesh) {
        if (t.lo[2] > z || t.hi[2] <= z) continue;  // half-open: a vertex on the plane counts as below
        const double* v[3] = {t.a, t.b, t.c};
        double pts[2][2];
        int count = 0;
        for (int i = 0; i < 3 && count < 2; ++i) {
            const double* p = v[i];
            const double* q = v[(i + 1) % 3];
            const bool p_above = p[2] > z, q_above = q[2] > z;
            if (p_above == q_above) continue;
            const double f = (z - p[2]) / (q[2] - p[2]);
            pts[count][0] = p[0] + f * (q[0] - p[0]);
            pts[count][1] = p[1] + f * (q[1] - p[1]);
            ++count;
        }
        if (count < 2) continue;
        // Counter-clockwise around the solid: direction z x n for the outward normal n.
        const double ux = t.b[0] - t.a[0], uy = t.b[1] - t.a[1], uz = t.b[2] - t.a[2];
        const double vx = t.c[0] - t.a[0], vy = t.c[1] - t.a[1], vz = t.c[2] - t.a[2];
        const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz;
        const double tx = -ny, ty = nx;
        const double dx = pts[1][0] - pts[0][0], dy = pts[1][1] - pts[0][1];
        if (dx * tx + dy * ty >= 0.0) {
            segments.push_back({pts[0][0], pts[0][1], pts[1][0], pts[1][1]});
        } else {
            segments.push_back({pts[1][0], pts[1][1], pts[0][0], pts[0][1]});
        }
    }
    return segments;
}

py::array_t<double> section_segments(const ArrF64& triangles, double z) {
    const std::vector<Triangle> mesh = load_triangles(triangles);
    const std::vector<Segment> segments = plane_section(mesh, z);
    py::array_t<double> result({static_cast<py::ssize_t>(segments.size()), static_cast<py::ssize_t>(2),
                                static_cast<py::ssize_t>(2)});
    double* out = result.mutable_data();
    for (std::size_t i = 0; i < segments.size(); ++i) {
        out[i * 4 + 0] = segments[i].x0;
        out[i * 4 + 1] = segments[i].y0;
        out[i * 4 + 2] = segments[i].x1;
        out[i * 4 + 3] = segments[i].y1;
    }
    return result;
}

// Exact per-cell area fraction of the section of one or more solids.
// ``triangle_sets`` is a list of (n, 3, 3) arrays; their coverages add (the
// solids of one layer touch but do not overlap), clamped to [0, 1].
py::array_t<double> plane_section_coverage(const std::vector<ArrF64>& triangle_sets, double z, double origin_x,
                                           double origin_y, double pitch, int rows, int cols) {
    if (rows < 1 || cols < 1 || pitch <= 0.0) throw std::invalid_argument("grid must be positive");
    const int stride = cols + 2;
    std::vector<double> acc(static_cast<std::size_t>(rows) * stride, 0.0);
    {
        py::gil_scoped_release release;
        for (const ArrF64& triangles : triangle_sets) {
            const std::vector<Triangle> mesh = load_triangles(triangles);
            for (const Segment& s : plane_section(mesh, z)) {
                accumulate_line(acc, rows, cols, (s.x0 - origin_x) / pitch, (s.y0 - origin_y) / pitch,
                                (s.x1 - origin_x) / pitch, (s.y1 - origin_y) / pitch);
            }
        }
    }
    py::array_t<double> result({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    double* out = result.mutable_data();
    for (int y = 0; y < rows; ++y) {
        double running = 0.0;
        const double* line = acc.data() + static_cast<std::size_t>(y) * stride;
        for (int x = 0; x < cols; ++x) {
            running += line[x];
            out[static_cast<std::size_t>(y) * cols + x] = std::min(1.0, std::fabs(running));
        }
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_voxelize_native, m) {
    m.doc() = "Generalized winding number point-in-solid tests for tessellated STEP solids";
    m.def("winding_numbers", &winding_numbers, py::arg("points"), py::arg("triangles"), py::arg("threads") = 1);
    m.def("contains", &contains, py::arg("points"), py::arg("triangles"), py::arg("threshold") = 0.5,
          py::arg("threads") = 1);
    m.def("section_segments", &section_segments, py::arg("triangles"), py::arg("z"));
    m.def("plane_section_coverage", &plane_section_coverage, py::arg("triangle_sets"), py::arg("z"),
          py::arg("origin_x"), py::arg("origin_y"), py::arg("pitch"), py::arg("rows"), py::arg("cols"));
}
