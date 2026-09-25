// Direct dipole superposition on the host: every observation point (or far
// field direction) sums the exact Hertzian-dipole fields of all sources.  The
// pairwise work is O(points x sources) with no pair matrix; threads own
// disjoint point ranges, so the result for a fixed source order is
// deterministic and independent of the thread count.  Formulas and the
// e^{+j omega t} convention match ``fields.py`` and ``far_field.py``.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

using c128 = std::complex<double>;
using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrC128 = py::array_t<c128, py::array::c_style | py::array::forcecast>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kMu0 = 1.25663706212e-6;
constexpr double kSpeedOfLight = 299792458.0;
constexpr double kEta = kMu0 * kSpeedOfLight;

void check_sources(const ArrF64& position, const ArrC128& moment, py::ssize_t& count) {
    if (position.ndim() != 2 || position.shape(1) != 3) {
        throw std::invalid_argument("source positions must have shape (sources, 3)");
    }
    if (moment.ndim() != 2 || moment.shape(0) != position.shape(0) || moment.shape(1) != 3) {
        throw std::invalid_argument("source moments must match the positions in shape");
    }
    count = position.shape(0);
}

}  // namespace

// Returns (H (P,3) complex128, E (P,3) complex128 or an empty array when
// ``electric`` is false).  Raises if a point coincides with a source.
py::tuple evaluate_fields(ArrF64 points, ArrF64 source_position, ArrC128 source_moment,
                          double wavenumber, bool electric, int threads) {
    py::ssize_t sources = 0;
    check_sources(source_position, source_moment, sources);
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (points, 3)");
    }
    if (electric && wavenumber == 0.0) {
        throw std::invalid_argument("the electric field is undefined at zero frequency");
    }
    const py::ssize_t count = points.shape(0);
    ArrC128 magnetic({count, static_cast<py::ssize_t>(3)});
    ArrC128 electric_field({electric ? count : 0, static_cast<py::ssize_t>(3)});
    const double* pts = points.data();
    const double* pos = source_position.data();
    const c128* mom = source_moment.data();
    c128* hout = magnetic.mutable_data();
    c128* eout = electric ? electric_field.mutable_data() : nullptr;
    bool coincident = false;
    {
        py::gil_scoped_release release;
        const double k = wavenumber;
        const c128 jk(0.0, k);
        const c128 h_front = c128(0.0, -k / (4.0 * kPi));
        const c128 e_front = kEta / (4.0 * kPi * jk);
#pragma omp parallel for schedule(static) num_threads(threads < 1 ? 1 : threads) if (threads > 1)
        for (py::ssize_t p = 0; p < count; ++p) {
            const double ox = pts[3 * p], oy = pts[3 * p + 1], oz = pts[3 * p + 2];
            c128 hx(0.0), hy(0.0), hz(0.0), ex(0.0), ey(0.0), ez(0.0);
            bool bad = false;
            for (py::ssize_t s = 0; s < sources; ++s) {
                const double rx = ox - pos[3 * s], ry = oy - pos[3 * s + 1], rz = oz - pos[3 * s + 2];
                const double r2 = rx * rx + ry * ry + rz * rz;
                if (r2 == 0.0) {
                    bad = true;
                    continue;
                }
                const double r = std::sqrt(r2);
                const double inv_r = 1.0 / r;
                const double nx = rx * inv_r, ny = ry * inv_r, nz = rz * inv_r;
                const c128 px = mom[3 * s], py_ = mom[3 * s + 1], pz = mom[3 * s + 2];
                // n x p
                const c128 cx = ny * pz - nz * py_;
                const c128 cy = nz * px - nx * pz;
                const c128 cz = nx * py_ - ny * px;
                c128 h_scale;
                c128 phase(1.0, 0.0);
                if (k == 0.0) {
                    h_scale = -(inv_r * inv_r) / (4.0 * kPi);
                } else {
                    double sn, cs;
                    sincos(-k * r, &sn, &cs);
                    phase = c128(cs, sn);
                    h_scale = h_front * (1.0 + 1.0 / (jk * r)) * phase * inv_r;
                }
                hx += cx * h_scale;
                hy += cy * h_scale;
                hz += cz * h_scale;
                if (eout != nullptr) {
                    const c128 radial = nx * px + ny * py_ + nz * pz;
                    // (n x p) x n
                    const c128 fx = cy * nz - cz * ny;
                    const c128 fy = cz * nx - cx * nz;
                    const c128 fz = cx * ny - cy * nx;
                    const double far = k * k * inv_r;
                    const c128 near = c128(inv_r * inv_r * inv_r, k * inv_r * inv_r);
                    const c128 e_scale = e_front * phase;
                    ex += (fx * far + (3.0 * nx * radial - px) * near) * e_scale;
                    ey += (fy * far + (3.0 * ny * radial - py_) * near) * e_scale;
                    ez += (fz * far + (3.0 * nz * radial - pz) * near) * e_scale;
                }
            }
            hout[3 * p] = hx;
            hout[3 * p + 1] = hy;
            hout[3 * p + 2] = hz;
            if (eout != nullptr) {
                eout[3 * p] = ex;
                eout[3 * p + 1] = ey;
                eout[3 * p + 2] = ez;
            }
            if (bad) {
#pragma omp atomic write
                coincident = true;
            }
        }
    }
    if (coincident) throw std::invalid_argument("an observation point coincides with a source");
    return py::make_tuple(magnetic, electric_field);
}

// Transverse phase-weighted sum per direction: F(n) = sum_i [(n x p_i) x n] e^{+jk n.r_i}.
ArrC128 far_field_pattern(ArrF64 directions, ArrF64 source_position, ArrC128 source_moment,
                          double wavenumber, int threads) {
    py::ssize_t sources = 0;
    check_sources(source_position, source_moment, sources);
    if (directions.ndim() != 2 || directions.shape(1) != 3) {
        throw std::invalid_argument("directions must have shape (directions, 3)");
    }
    const py::ssize_t count = directions.shape(0);
    ArrC128 pattern({count, static_cast<py::ssize_t>(3)});
    const double* dir = directions.data();
    const double* pos = source_position.data();
    const c128* mom = source_moment.data();
    c128* out = pattern.mutable_data();
    {
        py::gil_scoped_release release;
        const double k = wavenumber;
#pragma omp parallel for schedule(static) num_threads(threads < 1 ? 1 : threads) if (threads > 1)
        for (py::ssize_t d = 0; d < count; ++d) {
            const double nx = dir[3 * d], ny = dir[3 * d + 1], nz = dir[3 * d + 2];
            c128 wx(0.0), wy(0.0), wz(0.0);
            for (py::ssize_t s = 0; s < sources; ++s) {
                const double dot = nx * pos[3 * s] + ny * pos[3 * s + 1] + nz * pos[3 * s + 2];
                double sn, cs;
                sincos(k * dot, &sn, &cs);
                const c128 phase(cs, sn);
                wx += phase * mom[3 * s];
                wy += phase * mom[3 * s + 1];
                wz += phase * mom[3 * s + 2];
            }
            const c128 radial = nx * wx + ny * wy + nz * wz;
            out[3 * d] = wx - nx * radial;
            out[3 * d + 1] = wy - ny * radial;
            out[3 * d + 2] = wz - nz * radial;
        }
    }
    return pattern;
}

// Current elements of a sheet-PEEC solve: one element per branch, in the
// branch order of ``SheetMesh`` (x branches, y branches, via branches).  The
// geometry matches ``sources.dipoles_from_sheet_peec``: an in-plane element
// sits midway along its branch on the layer's height with moment ``I pitch``
// along the branch axis; a via element sits at the cell centre halfway between
// the two layers with moment ``I (z_upper - z_lower)`` along z.  Returns
// (positions (N,3) float64, moments (N,3) complex128).
py::tuple sheet_branch_dipoles(py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> branch_x,
                               py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> branch_y,
                               py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> vias,
                               double pitch, ArrF64 heights, ArrC128 current, int threads) {
    if (branch_x.ndim() != 2 || branch_x.shape(1) != 3 || branch_y.ndim() != 2 || branch_y.shape(1) != 3) {
        throw std::invalid_argument("branch_x and branch_y must have shape (branches, 3): layer, row, col");
    }
    if (vias.ndim() != 2 || vias.shape(1) != 4) {
        throw std::invalid_argument("vias must have shape (vias, 4): lower_layer, upper_layer, row, col");
    }
    const py::ssize_t nx = branch_x.shape(0), ny = branch_y.shape(0), nv = vias.shape(0);
    const py::ssize_t total = nx + ny + nv;
    if (current.ndim() != 1 || current.shape(0) != total) {
        throw std::invalid_argument("current must hold one value per branch");
    }
    const py::ssize_t layers = heights.size();
    const std::int64_t* bx = branch_x.data();
    const std::int64_t* by = branch_y.data();
    const std::int64_t* bv = vias.data();
    const double* z = heights.data();
    const c128* cur = current.data();
    for (py::ssize_t i = 0; i < nx; ++i) {
        if (bx[3 * i] < 0 || bx[3 * i] >= layers) throw std::invalid_argument("branch_x names a layer outside heights");
    }
    for (py::ssize_t i = 0; i < ny; ++i) {
        if (by[3 * i] < 0 || by[3 * i] >= layers) throw std::invalid_argument("branch_y names a layer outside heights");
    }
    for (py::ssize_t i = 0; i < nv; ++i) {
        if (bv[4 * i] < 0 || bv[4 * i] >= layers || bv[4 * i + 1] < 0 || bv[4 * i + 1] >= layers) {
            throw std::invalid_argument("a via names a layer outside heights");
        }
    }
    ArrF64 positions({total, static_cast<py::ssize_t>(3)});
    ArrC128 moments({total, static_cast<py::ssize_t>(3)});
    double* pos = positions.mutable_data();
    c128* mom = moments.mutable_data();
    {
        py::gil_scoped_release release;
        const int team = threads < 1 ? 1 : threads;
#pragma omp parallel for schedule(static) num_threads(team) if (team > 1 && total > 4096)
        for (py::ssize_t i = 0; i < total; ++i) {
            double* p = pos + 3 * i;
            c128* q = mom + 3 * i;
            if (i < nx) {
                const std::int64_t layer = bx[3 * i], row = bx[3 * i + 1], col = bx[3 * i + 2];
                p[0] = (static_cast<double>(col) + 1.0) * pitch;
                p[1] = (static_cast<double>(row) + 0.5) * pitch;
                p[2] = z[layer];
                q[0] = cur[i] * pitch;
                q[1] = c128(0.0, 0.0);
                q[2] = c128(0.0, 0.0);
            } else if (i < nx + ny) {
                const py::ssize_t k = i - nx;
                const std::int64_t layer = by[3 * k], row = by[3 * k + 1], col = by[3 * k + 2];
                p[0] = (static_cast<double>(col) + 0.5) * pitch;
                p[1] = (static_cast<double>(row) + 1.0) * pitch;
                p[2] = z[layer];
                q[0] = c128(0.0, 0.0);
                q[1] = cur[i] * pitch;
                q[2] = c128(0.0, 0.0);
            } else {
                const py::ssize_t k = i - nx - ny;
                const std::int64_t lower = bv[4 * k], upper = bv[4 * k + 1], row = bv[4 * k + 2], col = bv[4 * k + 3];
                p[0] = (static_cast<double>(col) + 0.5) * pitch;
                p[1] = (static_cast<double>(row) + 0.5) * pitch;
                p[2] = 0.5 * (z[lower] + z[upper]);
                q[0] = c128(0.0, 0.0);
                q[1] = c128(0.0, 0.0);
                q[2] = cur[i] * (z[upper] - z[lower]);
            }
        }
    }
    return py::make_tuple(positions, moments);
}

PYBIND11_MODULE(_dipole_native, m) {
    m.doc() = "Direct Hertzian-dipole superposition on the host";
    m.def("sheet_branch_dipoles", &sheet_branch_dipoles, py::arg("branch_x"), py::arg("branch_y"), py::arg("vias"),
          py::arg("pitch"), py::arg("heights"), py::arg("current"), py::arg("threads") = 1);
    m.def("evaluate_fields", &evaluate_fields, py::arg("points"), py::arg("source_position"),
          py::arg("source_moment"), py::arg("wavenumber"), py::arg("electric") = true,
          py::arg("threads") = 1);
    m.def("far_field_pattern", &far_field_pattern, py::arg("directions"), py::arg("source_position"),
          py::arg("source_moment"), py::arg("wavenumber"), py::arg("threads") = 1);
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}
