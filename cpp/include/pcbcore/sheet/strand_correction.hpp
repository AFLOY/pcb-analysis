// The edge-strand precorrection of a uniform sheet mesh, applied without
// storing it.
//
// Python's edge_strands cuts the in-plane branches along a conductor's edge
// into strands and corrects, for every pair with a strand in it within a few
// cells, the grid's partial inductance to the exact one.  On a straight edge
// those corrections repeat: they depend on the two branches' strand layouts
// (their "types"), their layers and their offset, not on where they are.  The
// strip pair at 0.1 mm held 7.6M entries but 2.7k distinct values.  This keeps
// one m x m block per (slot, layer pair, source type, target type) and finds
// a strand's neighbours through a grid lookup when it is applied, so the
// memory is the blocks plus a few integers per branch.
//
// One family of like-directed branches (one axis) is added at a time; in-plane
// branches of the two axes do not couple.
#pragma once

#include <cstdint>
#include <vector>

namespace pcbcore::sheet {

class StrandCorrection {
public:
    // ``size``: the length of the stranded branch vector the correction acts on.
    explicit StrandCorrection(std::int64_t size) : size_(size) {}

    // One axis.  ``lookup`` (planes x rows x cols): the family's local branch on
    // each cell or -1.  ``cells`` (count x 3): plane, row, col of each branch.
    // ``type`` (count): 0 for an uncut branch, its layout otherwise.
    // ``strand_start`` (count + 1) and ``strand_id``: the branch's strands, as
    // indices into the stranded vector (an uncut branch: itself).
    // ``sources``: the cut branches.  ``slot_row``/``slot_col`` (slots): the
    // offsets searched.  ``partner_start`` (planes + 1)/``partner``: the planes
    // each plane couples to.  ``block`` (slots x planes x planes x types x
    // types): the block of each combination or -1.  ``correction``/``exact``
    // (blocks x width x width): the blocks, row-major, source strand by target
    // strand, zero-padded to ``width``.
    void add_family(std::int64_t planes, std::int64_t rows, std::int64_t cols, std::vector<std::int32_t> lookup,
                    std::vector<std::int32_t> cells, std::vector<std::int32_t> type,
                    std::vector<std::int64_t> strand_start, std::vector<std::int64_t> strand_id,
                    std::vector<std::int32_t> sources, std::vector<std::int32_t> slot_row,
                    std::vector<std::int32_t> slot_col, std::vector<std::int32_t> partner_start,
                    std::vector<std::int32_t> partner, std::int32_t types, std::vector<std::int32_t> block,
                    std::int32_t width, std::vector<double> correction, std::vector<double> exact);

    // y = C x for a real x of the stranded length; y is overwritten.
    void apply(const double* x, double* y, int threads) const;

    // The entries of C (``exact`` false) or of the exact partial inductances
    // (true), as COO; every pair appears once in each direction.
    void entries(bool exact, std::vector<std::int64_t>& row, std::vector<std::int64_t>& col,
                 std::vector<double>& value) const;

    [[nodiscard]] std::int64_t size() const noexcept { return size_; }
    [[nodiscard]] std::int64_t stored_bytes() const noexcept;

private:
    struct Family {
        std::int64_t planes{0}, rows{0}, cols{0};
        std::vector<std::int32_t> lookup, cells, type;
        std::vector<std::int64_t> strand_start, strand_id;
        std::vector<std::int32_t> sources, slot_row, slot_col, partner_start, partner;
        std::int32_t types{0};
        std::vector<std::int32_t> block;
        std::int32_t width{0};
        std::vector<double> correction, exact;
    };

    template <typename Visit>
    void walk(const Family& family, std::int64_t source_index, Visit&& visit) const;

    std::int64_t size_{0};
    std::vector<Family> families_;
};

}  // namespace pcbcore::sheet
