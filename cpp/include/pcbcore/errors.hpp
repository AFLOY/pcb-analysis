// Error kinds of the pcbcore library and the Python exceptions they become.
//
// The bindings translate them so that callers keep catching what the NumPy
// implementations raised:
//   InvalidInput -> ValueError         (bad shapes, indices, arguments)
//   Singular     -> RuntimeError       (a factorisation met a zero pivot)
//   NonFinite    -> FloatingPointError (NaN or infinity where a value is needed)
// std::bad_alloc becomes MemoryError through pybind11's own translation.
#pragma once

#include <stdexcept>
#include <string>

namespace pcbcore {

class InvalidInput : public std::invalid_argument {
public:
    explicit InvalidInput(const std::string& what) : std::invalid_argument(what) {}
};

class Singular : public std::runtime_error {
public:
    explicit Singular(const std::string& what) : std::runtime_error(what) {}
};

class NonFinite : public std::runtime_error {
public:
    explicit NonFinite(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace pcbcore
