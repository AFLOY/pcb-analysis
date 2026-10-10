// One excitation of a sheet PEEC mesh: the DC nodal solve or the AC
// saddle-point solve by preconditioned GMRES.
//
// The mesh arrives as its incidence (branch ``b`` leaves node ``left[b]`` and
// enters ``right[b]``), branch resistances and the current injected at each
// node; the partial inductance as a FluxOperator over the full branch vector.
// The steps follow the NumPy implementation:
//
// * connected components of the copper; components no terminal drives are
//   left out (nodes and branches), each driven component must close its own
//   current, and the first node of each is grounded;
// * DC: the nodal system A^T R^-1 A by SuperLU;
// * AC: [[R + j w L, -A], [A^T, 0]] [I; V] = [0; J] by GMRES (SciPy's
//   algorithm) with the near (saddle LU), block (Z_near LU and a diagonal
//   Schur complement) or diagonal preconditioner.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pcbcore::sheet {

// Partial inductance applied to real branch currents in mesh branch order.
class FluxOperator {
public:
    virtual ~FluxOperator() = default;
    [[nodiscard]] virtual std::int64_t branch_count() const noexcept = 0;
    virtual void apply(const double* currents, double* flux, int threads) const = 0;
};

class ConvolutionOperator;
class PfftOperator;

// Scatter branch currents into an operator's per-family inputs and gather the
// fluxes back.  ``family[b]`` is 0 (x), 1 (y) or 2 (z); ``position[b]`` the
// flat index in that family's input.  A position written twice keeps the
// later branch, and both read the value back.
class ConvolutionFlux final : public FluxOperator {
public:
    ConvolutionFlux(std::shared_ptr<const ConvolutionOperator> op, std::vector<std::int8_t> family,
                    std::vector<std::int64_t> position, std::int64_t in_plane_size, std::int64_t vertical_size);
    [[nodiscard]] std::int64_t branch_count() const noexcept override {
        return static_cast<std::int64_t>(family_.size());
    }
    void apply(const double* currents, double* flux, int threads) const override;

private:
    std::shared_ptr<const ConvolutionOperator> op_;
    std::vector<std::int8_t> family_;
    std::vector<std::int64_t> position_;
    std::int64_t in_plane_size_;
    std::int64_t vertical_size_;
};

class PfftFlux final : public FluxOperator {
public:
    // ``family_ids`` are the operator's family indices for x, y and z (-1: absent).
    PfftFlux(std::shared_ptr<const PfftOperator> op, std::vector<std::int8_t> family,
             std::vector<std::int64_t> position, std::vector<std::int64_t> family_ids);
    [[nodiscard]] std::int64_t branch_count() const noexcept override {
        return static_cast<std::int64_t>(family_.size());
    }
    void apply(const double* currents, double* flux, int threads) const override;

private:
    std::shared_ptr<const PfftOperator> op_;
    std::vector<std::int8_t> family_;
    std::vector<std::int64_t> position_;
    std::vector<std::int64_t> family_ids_;
    std::vector<std::int64_t> sizes_;
};

struct NearInductance {  // CSR over mesh branches
    std::vector<std::int64_t> indptr;
    std::vector<std::int64_t> indices;
    std::vector<double> data;
};

struct SheetProblem {
    std::int64_t node_count{0};
    std::int64_t branch_count{0};
    const std::int64_t* left{nullptr};
    const std::int64_t* right{nullptr};
    const double* resistance{nullptr};
    const std::complex<double>* injection{nullptr};
    double frequency_hz{0.0};
    double tolerance{1e-10};
    std::int64_t max_iterations{400};
    std::int64_t restart{60};
    std::string preconditioner{"auto"};
    std::int64_t auto_block_from_unknowns{60000};
    const FluxOperator* flux{nullptr};
    const NearInductance* near{nullptr};     // required for near and block
    const double* self_inductance{nullptr};  // per branch, required for block and diagonal
};

struct SheetResult {
    std::vector<std::complex<double>> node_voltage;
    std::vector<std::complex<double>> branch_current;
    std::int64_t iterations{0};
    double residual{0.0};
    bool converged{false};
    std::int64_t grounded_node{0};
    std::int64_t undriven_nodes{0};
    std::string preconditioner;
};

[[nodiscard]] SheetResult solve_sheet(const SheetProblem& problem, int threads);

}  // namespace pcbcore::sheet
