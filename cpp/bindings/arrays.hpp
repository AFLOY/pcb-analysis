// NumPy conversions shared by the binding files of ``electrical._pcbcore``.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>
#include <vector>

namespace pcbcore_bindings {

namespace py = pybind11;

// An input array viewed without a copy when its dtype and layout match.
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

template <typename T>
py::array_t<T> copy_array(const std::vector<T>& values, std::vector<py::ssize_t> shape) {
    std::vector<T> copy(values);
    return to_array(std::move(copy), std::move(shape));
}

}  // namespace pcbcore_bindings
