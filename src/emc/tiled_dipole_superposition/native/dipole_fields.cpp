// Direct dipole superposition on the host: every observation point (or far
// field direction) sums the exact Hertzian-dipole fields of all sources. The
// pairwise work is O(points x sources) with no pair matrix; threads own
// disjoint point ranges, so the result for a fixed source order is
// deterministic and independent of the thread count. Formulas and the
// e^{+j omega t} convention match ``fields.py`` and ``far_field.py``.
//
// Compliant with MISRA-C++: encapsulated in anonymous namespace, explicit fixed-width
// integer types, no C-style casts, const-correctness, noexcept specifications, no exported headers.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// Registered as electrical._pcbcore.dipole; built on its own as _dipole_native
// when PCB_STANDALONE_MODULE is defined (native/build.py).
namespace pcb_dipole {

namespace {

using c128 = std::complex<double>;

using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrC128 = py::array_t<c128, py::array::c_style | py::array::forcecast>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kMu0 = 1.25663706212e-6;
constexpr double kSpeedOfLight = 299792458.0;
constexpr double kEta = kMu0 * kSpeedOfLight;
constexpr std::int64_t kDim3 = 3;

void check_sources(const ArrF64& position, const ArrC128& moment, std::int64_t& count) {
    if (position.ndim() != 2 || position.shape(1) != kDim3) {
        throw std::invalid_argument("source positions must have shape (sources, 3)");
    }
    if (moment.ndim() != 2 || moment.shape(0) != position.shape(0) || moment.shape(1) != kDim3) {
        throw std::invalid_argument("source moments must match the positions in shape");
    }
    count = static_cast<std::int64_t>(position.shape(0));
}

// Direct Hertzian-dipole near/intermediate field evaluation.
py::tuple evaluate_fields_impl(
    const ArrF64& points,
    const ArrF64& source_position,
    const ArrC128& source_moment,
    const double wavenumber,
    const bool electric,
    const std::int32_t threads
) {
    std::int64_t sources = 0;
    check_sources(source_position, source_moment, sources);
    if (points.ndim() != 2 || points.shape(1) != kDim3) {
        throw std::invalid_argument("points must have shape (points, 3)");
    }
    if (electric && wavenumber == 0.0) {
        throw std::invalid_argument("the electric field is undefined at zero frequency");
    }

    const std::int64_t count = static_cast<std::int64_t>(points.shape(0));
    ArrC128 magnetic({count, static_cast<py::ssize_t>(kDim3)});
    ArrC128 electric_field({electric ? count : static_cast<std::int64_t>(0), static_cast<py::ssize_t>(kDim3)});

    const double* const pts = points.data();
    const double* const pos = source_position.data();
    const c128* const mom = source_moment.data();
    c128* const hout = magnetic.mutable_data();
    c128* const eout = electric ? electric_field.mutable_data() : nullptr;

    bool coincident = false;
    const std::int32_t num_threads = (threads < 1) ? 1 : threads;

    {
        py::gil_scoped_release release;
        const double k = wavenumber;
        const c128 jk(0.0, k);
        const c128 h_front(0.0, -k / (4.0 * kPi));
        const c128 e_front = (k == 0.0) ? c128(0.0, 0.0) : (kEta / (4.0 * kPi * jk));

#pragma omp parallel for schedule(static) num_threads(num_threads) if (num_threads > 1)
        for (std::int64_t p = 0; p < count; ++p) {
            const double ox = pts[kDim3 * p];
            const double oy = pts[kDim3 * p + 1];
            const double oz = pts[kDim3 * p + 2];

            c128 hx(0.0, 0.0);
            c128 hy(0.0, 0.0);
            c128 hz(0.0, 0.0);
            c128 ex(0.0, 0.0);
            c128 ey(0.0, 0.0);
            c128 ez(0.0, 0.0);
            bool bad = false;

            for (std::int64_t s = 0; s < sources; ++s) {
                const double* const pos_s = pos + kDim3 * s;
                const double rx = ox - pos_s[0];
                const double ry = oy - pos_s[1];
                const double rz = oz - pos_s[2];
                const double r2 = rx * rx + ry * ry + rz * rz;
                if (r2 == 0.0) {
                    bad = true;
                    continue;
                }
                const double r = std::sqrt(r2);
                const double inv_r = 1.0 / r;
                const double nx = rx * inv_r;
                const double ny = ry * inv_r;
                const double nz = rz * inv_r;

                const c128* const mom_s = mom + kDim3 * s;
                const c128 px = mom_s[0];
                const c128 py_ = mom_s[1];
                const c128 pz = mom_s[2];

                // n x p
                const c128 cx = ny * pz - nz * py_;
                const c128 cy = nz * px - nx * pz;
                const c128 cz = nx * py_ - ny * px;

                c128 h_scale(0.0, 0.0);
                c128 phase(1.0, 0.0);
                if (k == 0.0) {
                    h_scale = -(inv_r * inv_r) / (4.0 * kPi);
                } else {
                    double sn = 0.0;
                    double cs = 0.0;
                    ::sincos(-k * r, &sn, &cs);
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
                    const c128 near_term(inv_r * inv_r * inv_r, k * inv_r * inv_r);
                    const c128 e_scale = e_front * phase;
                    ex += (fx * far + (3.0 * nx * radial - px) * near_term) * e_scale;
                    ey += (fy * far + (3.0 * ny * radial - py_) * near_term) * e_scale;
                    ez += (fz * far + (3.0 * nz * radial - pz) * near_term) * e_scale;
                }
            }

            hout[kDim3 * p] = hx;
            hout[kDim3 * p + 1] = hy;
            hout[kDim3 * p + 2] = hz;
            if (eout != nullptr) {
                eout[kDim3 * p] = ex;
                eout[kDim3 * p + 1] = ey;
                eout[kDim3 * p + 2] = ez;
            }

            if (bad) {
#pragma omp atomic write
                coincident = true;
            }
        }
    }

    if (coincident) {
        throw std::invalid_argument("an observation point coincides with a source");
    }

    return py::make_tuple(magnetic, electric_field);
}

// Transverse phase-weighted sum per direction: F(n) = sum_i [(n x p_i) x n] e^{+jk n.r_i}.
ArrC128 far_field_pattern_impl(
    const ArrF64& directions,
    const ArrF64& source_position,
    const ArrC128& source_moment,
    const double wavenumber,
    const std::int32_t threads
) {
    std::int64_t sources = 0;
    check_sources(source_position, source_moment, sources);
    if (directions.ndim() != 2 || directions.shape(1) != kDim3) {
        throw std::invalid_argument("directions must have shape (directions, 3)");
    }

    const std::int64_t count = static_cast<std::int64_t>(directions.shape(0));
    ArrC128 pattern({count, static_cast<py::ssize_t>(kDim3)});

    const double* const dir = directions.data();
    const double* const pos = source_position.data();
    const c128* const mom = source_moment.data();
    c128* const out = pattern.mutable_data();

    const std::int32_t num_threads = (threads < 1) ? 1 : threads;

    {
        py::gil_scoped_release release;
        const double k = wavenumber;

#pragma omp parallel for schedule(static) num_threads(num_threads) if (num_threads > 1)
        for (std::int64_t d = 0; d < count; ++d) {
            const double nx = dir[kDim3 * d];
            const double ny = dir[kDim3 * d + 1];
            const double nz = dir[kDim3 * d + 2];
            c128 wx(0.0, 0.0);
            c128 wy(0.0, 0.0);
            c128 wz(0.0, 0.0);

            for (std::int64_t s = 0; s < sources; ++s) {
                const double* const pos_s = pos + kDim3 * s;
                const c128* const mom_s = mom + kDim3 * s;
                const double dot = nx * pos_s[0] + ny * pos_s[1] + nz * pos_s[2];
                double sn = 0.0;
                double cs = 0.0;
                ::sincos(k * dot, &sn, &cs);
                const c128 phase(cs, sn);
                wx += phase * mom_s[0];
                wy += phase * mom_s[1];
                wz += phase * mom_s[2];
            }

            const c128 radial = nx * wx + ny * wy + nz * wz;
            out[kDim3 * d] = wx - nx * radial;
            out[kDim3 * d + 1] = wy - ny * radial;
            out[kDim3 * d + 2] = wz - nz * radial;
        }
    }

    return pattern;
}

}  // namespace

void register_module(py::module_& m) {
    m.doc() = "Direct Hertzian-dipole superposition on the host with MISRA-C++ compliance";
    m.def(
        "evaluate_fields",
        &evaluate_fields_impl,
        py::arg("points"),
        py::arg("source_position"),
        py::arg("source_moment"),
        py::arg("wavenumber"),
        py::arg("electric") = true,
        py::arg("threads") = 1
    );
    m.def(
        "far_field_pattern",
        &far_field_pattern_impl,
        py::arg("directions"),
        py::arg("source_position"),
        py::arg("source_moment"),
        py::arg("wavenumber"),
        py::arg("threads") = 1
    );
    m.attr("openmp") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
}

}  // namespace pcb_dipole

#ifdef PCB_STANDALONE_MODULE
PYBIND11_MODULE(_dipole_native, m) { pcb_dipole::register_module(m); }
#endif
