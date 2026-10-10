// The pcbcore extension module, ``electrical._pcbcore``.
//
// Bindings only convert: NumPy arrays in (viewed without a copy when their
// dtype and layout already match), the C++ call with the GIL released, NumPy
// arrays out (moved, not copied).  The Python facades wrap the results in the
// package's dataclasses.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pcbcore/errors.hpp"
#include "pcbcore/network/dc_network.hpp"
#include "pcbcore/sheet/convolution_operator.hpp"
#include "pcbcore/sheet/hoer_love.hpp"
#include "pcbcore/sheet/near_field.hpp"
#include "pcbcore/sheet/pfft_operator.hpp"
#include "pcbcore/sheet/sheet_solve.hpp"

namespace py = pybind11;

namespace {

template <typename T>
using Input = py::array_t<T, py::array::c_style | py::array::forcecast>;

// A NumPy array that owns ``values``: the vector moves into a capsule.
template <typename T>
py::array_t<T> to_array(std::vector<T>&& values, std::vector<py::ssize_t> shape) {
    auto* owner = new std::vector<T>(std::move(values));
    py::capsule release(owner, [](void* pointer) { delete static_cast<std::vector<T>*>(pointer); });
    return py::array_t<T>(std::move(shape), owner->data(), release);
}

template <typename T>
py::array_t<T> to_array(std::vector<T>&& values) {
    const auto size = static_cast<py::ssize_t>(values.size());
    return to_array(std::move(values), {size});
}

pcbcore::network::ConductanceNetworkView network_view(const std::int64_t node_count,
                                                      const Input<std::int64_t>& left,
                                                      const Input<std::int64_t>& right,
                                                      const Input<double>& conductance) {
    if (left.ndim() != 1 || right.ndim() != 1 || conductance.ndim() != 1 || left.size() != right.size() ||
        left.size() != conductance.size()) {
        throw pcbcore::InvalidInput("left, right and conductance must be 1-D and equally long");
    }
    return {node_count, static_cast<std::int64_t>(left.size()), left.data(), right.data(), conductance.data()};
}

py::dict solve_conductance_network(const std::int64_t node_count, const Input<std::int64_t>& left,
                                   const Input<std::int64_t>& right, const Input<double>& conductance,
                                   const std::int64_t reference, const Input<double>& injection,
                                   const Input<double>& weights) {
    const auto network = network_view(node_count, left, right, conductance);
    if (injection.ndim() != 1 || injection.size() != node_count) {
        throw pcbcore::InvalidInput("injection must have one entry per node");
    }
    if (weights.ndim() != 2 || weights.shape(1) != node_count) {
        throw pcbcore::InvalidInput("objective_weights must be objectives x node_count");
    }
    const std::int64_t objectives = weights.shape(0);
    pcbcore::network::DCNetworkResult result;
    {
        py::gil_scoped_release release;
        result = pcbcore::network::solve_conductance_network(network, reference, injection.data(),
                                                             objectives ? weights.data() : nullptr, objectives);
    }
    py::dict out;
    out["voltage_unknowns"] = to_array(std::move(result.voltage_unknowns));
    out["node_voltage"] = to_array(std::move(result.node_voltage));
    out["edge_current"] = to_array(std::move(result.edge_current));
    out["node_current"] = to_array(std::move(result.node_current));
    out["adjoint_voltage"] =
        to_array(std::move(result.adjoint_voltage), {static_cast<py::ssize_t>(objectives), static_cast<py::ssize_t>(node_count)});
    out["loss_w"] = result.loss_w;
    out["relative_residual"] = result.relative_residual;
    out["singular"] = result.singular;
    return out;
}

py::tuple split_branch_sensitivity(const std::int64_t node_count, const Input<std::int64_t>& left,
                                   const Input<std::int64_t>& right, const Input<double>& conductance,
                                   const Input<double>& branch_product, const Input<bool>& in_plane) {
    const auto network = network_view(node_count, left, right, conductance);
    if (branch_product.ndim() != 1 || in_plane.ndim() != 1 || branch_product.size() != left.size() ||
        in_plane.size() != left.size()) {
        throw pcbcore::InvalidInput("branch_product and in_plane must have one entry per branch");
    }
    pcbcore::network::BranchSensitivity result;
    {
        py::gil_scoped_release release;
        result = pcbcore::network::split_branch_sensitivity(network, branch_product.data(), in_plane.data());
    }
    return py::make_tuple(to_array(std::move(result.node_sensitivity)),
                          to_array(std::move(result.branch_sensitivity)), result.vertical_total);
}

// Kernel tables stacked as (pairs, padded_rows, padded_cols).
const double* stacked_tables(const Input<double>& tables, const std::int64_t pairs, const std::int64_t rows,
                             const std::int64_t cols, const char* name) {
    if (pairs == 0) {
        return nullptr;
    }
    if (tables.ndim() != 3 || tables.shape(0) != pairs || tables.shape(1) != 2 * rows || tables.shape(2) != 2 * cols) {
        throw pcbcore::InvalidInput(std::string(name) + " must be (pairs, 2 rows, 2 cols)");
    }
    return tables.data();
}

std::shared_ptr<pcbcore::sheet::ConvolutionOperator> make_convolution_operator(
    const std::int64_t layers, const std::int64_t rows, const std::int64_t cols, const std::int64_t levels,
    const Input<double>& tables_x, const Input<double>& tables_y, const Input<double>& tables_z, const int threads) {
    const std::int64_t pairs = layers * (layers + 1) / 2;
    const std::int64_t vertical = levels * (levels + 1) / 2;
    const double* x = stacked_tables(tables_x, pairs, rows, cols, "tables_x");
    const double* y = stacked_tables(tables_y, pairs, rows, cols, "tables_y");
    const double* z = stacked_tables(tables_z, vertical, rows, cols, "tables_z");
    py::gil_scoped_release release;
    return std::make_shared<pcbcore::sheet::ConvolutionOperator>(layers, rows, cols, levels, x, y, z, threads);
}

py::tuple apply_convolution(const pcbcore::sheet::ConvolutionOperator& op, const Input<double>& currents_x,
                            const Input<double>& currents_y, const std::optional<Input<double>>& currents_z,
                            const int threads) {
    const std::int64_t layers = op.layers();
    if (currents_x.ndim() != 3 || currents_y.ndim() != 3 || currents_x.shape(0) != layers ||
        currents_y.shape(0) != layers || currents_x.size() != currents_y.size()) {
        throw pcbcore::InvalidInput("currents_x and currents_y must be (layers, rows, cols)");
    }
    std::vector<py::ssize_t> shape{currents_x.shape(0), currents_x.shape(1), currents_x.shape(2)};
    std::vector<double> flux_x(static_cast<std::size_t>(currents_x.size()));
    std::vector<double> flux_y(static_cast<std::size_t>(currents_y.size()));
    std::vector<double> flux_z;
    const double* z = nullptr;
    if (currents_z.has_value() && op.levels() > 0) {
        if (currents_z->ndim() != 3 || currents_z->shape(0) != op.levels() || currents_z->shape(1) != shape[1] ||
            currents_z->shape(2) != shape[2]) {
            throw pcbcore::InvalidInput("currents_z must be (levels, rows, cols)");
        }
        flux_z.assign(static_cast<std::size_t>(currents_z->size()), 0.0);
        z = currents_z->data();
    }
    {
        py::gil_scoped_release release;
        op.apply(currents_x.data(), currents_y.data(), z, flux_x.data(), flux_y.data(),
                 z != nullptr ? flux_z.data() : nullptr, threads);
    }
    py::object out_z = py::none();
    if (z != nullptr) {
        out_z = to_array(std::move(flux_z), {op.levels(), shape[1], shape[2]});
    }
    return py::make_tuple(to_array(std::move(flux_x), shape), to_array(std::move(flux_y), shape), out_z);
}

py::array_t<double> build_kernel(const std::int64_t rows, const std::int64_t cols, const double length,
                                 const double width, const double thickness_a, const double thickness_b,
                                 const double separation, const bool along_x, const int near_radius_cells,
                                 const int threads) {
    std::vector<double> table;
    {
        py::gil_scoped_release release;
        table = pcbcore::sheet::build_kernel(rows, cols, {length, width, thickness_a}, {length, width, thickness_b},
                                             separation, along_x, near_radius_cells, threads);
    }
    return to_array(std::move(table), {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
}

py::array_t<double> build_vertical_kernel(const std::int64_t rows, const std::int64_t cols, const double pitch,
                                          const double span_a, const double span_b, const double center_separation,
                                          const int near_radius_cells, const int threads) {
    std::vector<double> table;
    {
        py::gil_scoped_release release;
        table = pcbcore::sheet::build_vertical_kernel(rows, cols, pitch, span_a, span_b, center_separation,
                                                      near_radius_cells, threads);
    }
    return to_array(std::move(table), {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
}

// Nine equally long 1-D arrays: bar a (length, width, thickness), bar b, offset.
py::tuple closed_form_arrays(const Input<double>& la, const Input<double>& wa, const Input<double>& ta,
                             const Input<double>& lb, const Input<double>& wb, const Input<double>& tb,
                             const Input<double>& du, const Input<double>& dv, const Input<double>& dw,
                             const int threads) {
    const py::ssize_t count = la.size();
    for (const auto* array : {&la, &wa, &ta, &lb, &wb, &tb, &du, &dv, &dw}) {
        if (array->ndim() != 1 || array->size() != count) {
            throw pcbcore::InvalidInput("closed_form_arrays takes nine 1-D arrays of one length");
        }
    }
    std::vector<double> out(static_cast<std::size_t>(count));
    double retained = 1.0;
    {
        py::gil_scoped_release release;
        const pcbcore::sheet::BarArrays a{la.data(), wa.data(), ta.data(), 1, 1, 1};
        const pcbcore::sheet::BarArrays b{lb.data(), wb.data(), tb.data(), 1, 1, 1};
        const pcbcore::sheet::OffsetArrays offset{du.data(), dv.data(), dw.data(), 1, 1, 1};
        retained = pcbcore::sheet::closed_form_arrays(count, a, b, offset, out.data(), threads);
    }
    return py::make_tuple(to_array(std::move(out)), retained);
}

const std::int64_t* cell_triples(const Input<std::int64_t>& cells, const char* name) {
    if (cells.size() == 0) {
        return nullptr;
    }
    if (cells.ndim() != 2 || cells.shape(1) != 3) {
        throw pcbcore::InvalidInput(std::string(name) + " must be (count, 3)");
    }
    return cells.data();
}

py::tuple uniform_near_field(const std::int64_t layers, const std::int64_t rows, const std::int64_t cols,
                             const std::int64_t levels, const int radius, const Input<double>& tables_x,
                             const Input<double>& tables_y, const Input<double>& tables_z,
                             const Input<std::int64_t>& branch_x, const Input<std::int64_t>& branch_y,
                             const Input<std::int64_t>& vias) {
    const std::int64_t pairs = layers * (layers + 1) / 2;
    const std::int64_t vertical = levels * (levels + 1) / 2;
    const double* x = stacked_tables(tables_x, pairs, rows, cols, "tables_x");
    const double* y = stacked_tables(tables_y, pairs, rows, cols, "tables_y");
    const double* z = stacked_tables(tables_z, vertical, rows, cols, "tables_z");
    const std::int64_t* bx = cell_triples(branch_x, "branch_x");
    const std::int64_t* by = cell_triples(branch_y, "branch_y");
    const std::int64_t* bz = cell_triples(vias, "vias");
    pcbcore::sheet::NearFieldEntries entries;
    {
        py::gil_scoped_release release;
        entries = pcbcore::sheet::uniform_near_field(layers, rows, cols, levels, radius, x, y, z, bx,
                                                     bx ? branch_x.shape(0) : 0, by, by ? branch_y.shape(0) : 0,
                                                     bz, bz ? vias.shape(0) : 0);
    }
    return py::make_tuple(to_array(std::move(entries.row)), to_array(std::move(entries.col)),
                          to_array(std::move(entries.value)));
}

pcbcore::sheet::CsrMatrix csr_from(const std::int64_t rows, const std::int64_t cols, const Input<std::int64_t>& indptr,
                                   const Input<std::int64_t>& indices, const Input<double>& data) {
    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1 || indices.size() != data.size() ||
        indptr.size() != rows + 1) {
        throw pcbcore::InvalidInput("inconsistent CSR arrays");
    }
    pcbcore::sheet::CsrMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.indptr.assign(indptr.data(), indptr.data() + indptr.size());
    m.indices.assign(indices.data(), indices.data() + indices.size());
    m.data.assign(data.data(), data.data() + data.size());
    return m;
}

std::int64_t pfft_add_family(pcbcore::sheet::PfftOperator& op, const std::int64_t planes, const std::int64_t branches,
                             const Input<std::int64_t>& p_indptr, const Input<std::int64_t>& p_indices,
                             const Input<double>& p_data, const Input<double>& weights,
                             const Input<std::int64_t>& kernel_of, const Input<std::int64_t>& c_indptr,
                             const Input<std::int64_t>& c_indices, const Input<double>& c_data, const std::int64_t nodes) {
    auto projection = csr_from(branches, nodes, p_indptr, p_indices, p_data);
    auto correction = csr_from(planes * branches, planes * branches, c_indptr, c_indices, c_data);
    std::vector<double> w(weights.data(), weights.data() + weights.size());
    std::vector<std::int64_t> k(kernel_of.data(), kernel_of.data() + kernel_of.size());
    return op.add_family(planes, std::move(projection), std::move(w), std::move(k), std::move(correction));
}

py::array_t<double> pfft_apply(const pcbcore::sheet::PfftOperator& op, const std::int64_t family,
                               const Input<double>& currents, const int threads) {
    const std::int64_t planes = op.planes(family);
    const std::int64_t n = op.branches(family);
    if (currents.size() != planes * n) {
        throw pcbcore::InvalidInput("currents must be planes x branches of the family");
    }
    std::vector<double> flux(static_cast<std::size_t>(planes * n), 0.0);
    {
        py::gil_scoped_release release;
        op.apply(family, currents.data(), flux.data(), threads);
    }
    return to_array(std::move(flux), {static_cast<py::ssize_t>(planes), static_cast<py::ssize_t>(n)});
}

// One excitation: ``op`` is a ConvolutionOperator or a PfftOperator (or None
// for DC); ``family``/``position`` place each mesh branch in its inputs.
py::dict solve_sheet(const std::int64_t node_count, const Input<std::int64_t>& left, const Input<std::int64_t>& right,
                     const Input<double>& resistance, const Input<std::complex<double>>& injection,
                     const double frequency_hz, const double tolerance, const std::int64_t max_iterations,
                     const std::int64_t restart, const std::string& preconditioner,
                     const std::int64_t auto_block_from_unknowns, const py::object& op,
                     const Input<std::int8_t>& family, const Input<std::int64_t>& position,
                     const Input<std::int64_t>& family_sizes, const std::optional<Input<std::int64_t>>& near_indptr,
                     const std::optional<Input<std::int64_t>>& near_indices,
                     const std::optional<Input<double>>& near_data,
                     const std::optional<Input<double>>& self_inductance, const int threads) {
    const std::int64_t branches = left.size();
    if (right.size() != branches || resistance.size() != branches || injection.size() != node_count) {
        throw pcbcore::InvalidInput("left, right, resistance must have one entry per branch and injection per node");
    }
    std::unique_ptr<pcbcore::sheet::FluxOperator> flux;
    if (!op.is_none()) {
        std::vector<std::int8_t> fam(family.data(), family.data() + family.size());
        std::vector<std::int64_t> pos(position.data(), position.data() + position.size());
        if (static_cast<std::int64_t>(fam.size()) != branches || family_sizes.size() != 3) {
            throw pcbcore::InvalidInput("family/position need one entry per branch and family_sizes three");
        }
        const std::int64_t* sizes = family_sizes.data();
        if (py::isinstance<pcbcore::sheet::ConvolutionOperator>(op)) {
            flux = std::make_unique<pcbcore::sheet::ConvolutionFlux>(
                op.cast<std::shared_ptr<pcbcore::sheet::ConvolutionOperator>>(), std::move(fam), std::move(pos),
                sizes[0], sizes[2]);
        } else if (py::isinstance<pcbcore::sheet::PfftOperator>(op)) {
            flux = std::make_unique<pcbcore::sheet::PfftFlux>(
                op.cast<std::shared_ptr<pcbcore::sheet::PfftOperator>>(), std::move(fam), std::move(pos),
                std::vector<std::int64_t>(sizes, sizes + 3));
        } else {
            throw pcbcore::InvalidInput("op must be a ConvolutionOperator or a PfftOperator");
        }
    }
    pcbcore::sheet::NearInductance near;
    const bool has_near = near_indptr.has_value() && near_indices.has_value() && near_data.has_value();
    if (has_near) {
        if (near_indptr->size() != branches + 1 || near_indices->size() != near_data->size()) {
            throw pcbcore::InvalidInput("near inductance must be a branch_count square CSR matrix");
        }
        near.indptr.assign(near_indptr->data(), near_indptr->data() + near_indptr->size());
        near.indices.assign(near_indices->data(), near_indices->data() + near_indices->size());
        near.data.assign(near_data->data(), near_data->data() + near_data->size());
    }
    if (self_inductance.has_value() && self_inductance->size() != branches) {
        throw pcbcore::InvalidInput("self_inductance needs one entry per branch");
    }
    pcbcore::sheet::SheetProblem problem;
    problem.node_count = node_count;
    problem.branch_count = branches;
    problem.left = left.data();
    problem.right = right.data();
    problem.resistance = resistance.data();
    problem.injection = injection.data();
    problem.frequency_hz = frequency_hz;
    problem.tolerance = tolerance;
    problem.max_iterations = max_iterations;
    problem.restart = restart;
    problem.preconditioner = preconditioner;
    problem.auto_block_from_unknowns = auto_block_from_unknowns;
    problem.flux = flux.get();
    problem.near = has_near ? &near : nullptr;
    problem.self_inductance = self_inductance.has_value() ? self_inductance->data() : nullptr;
    pcbcore::sheet::SheetResult result;
    {
        py::gil_scoped_release release;
        result = pcbcore::sheet::solve_sheet(problem, threads);
    }
    py::dict out;
    out["node_voltage"] = to_array(std::move(result.node_voltage));
    out["branch_current"] = to_array(std::move(result.branch_current));
    out["iterations"] = result.iterations;
    out["residual"] = result.residual;
    out["converged"] = result.converged;
    out["grounded_node"] = result.grounded_node;
    out["undriven_nodes"] = result.undriven_nodes;
    out["preconditioner"] = result.preconditioner;
    return out;
}

}  // namespace

PYBIND11_MODULE(_pcbcore, m) {
    m.doc() = "pcb-analysis C++ core: solvers and their kernels";

    // pybind11 already maps std::invalid_argument (InvalidInput) to ValueError,
    // std::runtime_error (Singular) to RuntimeError and std::bad_alloc to
    // MemoryError; NonFinite needs its own class.
    py::register_exception_translator([](std::exception_ptr pointer) {
        try {
            if (pointer) {
                std::rethrow_exception(pointer);
            }
        } catch (const pcbcore::NonFinite& error) {
            PyErr_SetString(PyExc_FloatingPointError, error.what());
        }
    });

    py::module_ network = m.def_submodule("network", "Conductance networks: the zero-frequency sheet mesh");
    network.def("solve_conductance_network", &solve_conductance_network, py::arg("node_count"), py::arg("left"),
                py::arg("right"), py::arg("conductance"), py::arg("reference"), py::arg("injection"),
                py::arg("weights"));
    network.def("split_branch_sensitivity", &split_branch_sensitivity, py::arg("node_count"), py::arg("left"),
                py::arg("right"), py::arg("conductance"), py::arg("branch_product"), py::arg("in_plane"));

    py::module_ sheet = m.def_submodule("sheet", "Sheet PEEC: operators, preconditioners and solves");
    py::class_<pcbcore::sheet::ConvolutionOperator, std::shared_ptr<pcbcore::sheet::ConvolutionOperator>>(sheet, "ConvolutionOperator")
        .def(py::init(&make_convolution_operator), py::arg("layers"), py::arg("rows"), py::arg("cols"),
             py::arg("levels"), py::arg("tables_x"), py::arg("tables_y"), py::arg("tables_z"), py::arg("threads"))
        .def("apply", &apply_convolution, py::arg("currents_x"), py::arg("currents_y"),
             py::arg("currents_z") = py::none(), py::arg("threads"))
        .def_property_readonly("spectrum_bytes", &pcbcore::sheet::ConvolutionOperator::spectrum_bytes);
    sheet.def("build_kernel", &build_kernel, py::arg("rows"), py::arg("cols"), py::arg("length"), py::arg("width"),
              py::arg("thickness_a"), py::arg("thickness_b"), py::arg("separation"), py::arg("along_x"),
              py::arg("near_radius_cells"), py::arg("threads"));
    sheet.def("build_vertical_kernel", &build_vertical_kernel, py::arg("rows"), py::arg("cols"), py::arg("pitch"),
              py::arg("span_a"), py::arg("span_b"), py::arg("center_separation"), py::arg("near_radius_cells"),
              py::arg("threads"));
    py::class_<pcbcore::sheet::PfftOperator, std::shared_ptr<pcbcore::sheet::PfftOperator>>(sheet, "PfftOperator")
        .def(py::init<std::int64_t, std::int64_t>(), py::arg("nodes_y"), py::arg("nodes_x"))
        .def(
            "add_kernel",
            [](pcbcore::sheet::PfftOperator& op, const Input<double>& table, const int threads) {
                return op.add_kernel(table.data(), threads);
            },
            py::arg("table"), py::arg("threads"))
        .def("add_family", &pfft_add_family, py::arg("planes"), py::arg("branches"), py::arg("projection_indptr"),
             py::arg("projection_indices"), py::arg("projection_data"), py::arg("weights"), py::arg("kernel_of"),
             py::arg("correction_indptr"), py::arg("correction_indices"), py::arg("correction_data"),
             py::arg("nodes"))
        .def("apply", &pfft_apply, py::arg("family"), py::arg("currents"), py::arg("threads"))
        .def_property_readonly("bytes", &pcbcore::sheet::PfftOperator::bytes);
    sheet.def("solve_sheet", &solve_sheet, py::arg("node_count"), py::arg("left"), py::arg("right"),
              py::arg("resistance"), py::arg("injection"), py::arg("frequency_hz"), py::arg("tolerance"),
              py::arg("max_iterations"), py::arg("restart"), py::arg("preconditioner"),
              py::arg("auto_block_from_unknowns"), py::arg("op"), py::arg("family"), py::arg("position"),
              py::arg("family_sizes"), py::arg("near_indptr") = py::none(), py::arg("near_indices") = py::none(),
              py::arg("near_data") = py::none(), py::arg("self_inductance") = py::none(), py::arg("threads"));
    sheet.def("uniform_near_field", &uniform_near_field, py::arg("layers"), py::arg("rows"), py::arg("cols"),
              py::arg("levels"), py::arg("radius"), py::arg("tables_x"), py::arg("tables_y"), py::arg("tables_z"),
              py::arg("branch_x"), py::arg("branch_y"), py::arg("vias"));
    sheet.def("closed_form_arrays", &closed_form_arrays, py::arg("length_a"), py::arg("width_a"),
              py::arg("thickness_a"), py::arg("length_b"), py::arg("width_b"), py::arg("thickness_b"),
              py::arg("du"), py::arg("dv"), py::arg("dw"), py::arg("threads"));
}
