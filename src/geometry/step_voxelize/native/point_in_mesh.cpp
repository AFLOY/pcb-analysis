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

}  // namespace

PYBIND11_MODULE(_voxelize_native, m) {
    m.doc() = "Generalized winding number point-in-solid tests for tessellated STEP solids";
    m.def("winding_numbers", &winding_numbers, py::arg("points"), py::arg("triangles"), py::arg("threads") = 1);
    m.def("contains", &contains, py::arg("points"), py::arg("triangles"), py::arg("threshold") = 0.5,
          py::arg("threads") = 1);
}
