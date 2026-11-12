#include "pcbcore/sheet/strand_correction.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "pcbcore/errors.hpp"

namespace pcbcore::sheet {

namespace {
using std::size_t;
}  // namespace

void StrandCorrection::add_family(const std::int64_t planes, const std::int64_t rows, const std::int64_t cols,
                                  std::vector<std::int32_t> lookup, std::vector<std::int32_t> cells,
                                  std::vector<std::int32_t> type, std::vector<std::int64_t> strand_start,
                                  std::vector<std::int64_t> strand_id, std::vector<std::int32_t> sources,
                                  std::vector<std::int32_t> slot_row, std::vector<std::int32_t> slot_col,
                                  std::vector<std::int32_t> partner_start, std::vector<std::int32_t> partner,
                                  const std::int32_t types, std::vector<std::int32_t> block, const std::int32_t width,
                                  std::vector<double> correction, std::vector<double> exact) {
    const auto count = static_cast<std::int64_t>(type.size());
    const auto slots = static_cast<std::int64_t>(slot_row.size());
    if (planes < 1 || rows < 1 || cols < 1 || static_cast<std::int64_t>(lookup.size()) != planes * rows * cols) {
        throw InvalidInput("lookup must be planes x rows x cols");
    }
    if (static_cast<std::int64_t>(cells.size()) != 3 * count ||
        static_cast<std::int64_t>(strand_start.size()) != count + 1 ||
        static_cast<std::int64_t>(strand_id.size()) != strand_start.back() || slot_col.size() != slot_row.size() ||
        static_cast<std::int64_t>(partner_start.size()) != planes + 1 ||
        static_cast<std::int64_t>(partner.size()) != partner_start.back()) {
        throw InvalidInput("inconsistent strand correction arrays");
    }
    if (types < 1 || width < 1 ||
        static_cast<std::int64_t>(block.size()) != slots * planes * planes * types * types ||
        correction.size() != exact.size() || correction.size() % static_cast<size_t>(width * width) != 0) {
        throw InvalidInput("inconsistent strand correction blocks");
    }
    const auto blocks = static_cast<std::int32_t>(correction.size() / static_cast<size_t>(width * width));
    for (const std::int32_t b : block) {
        if (b < -1 || b >= blocks) {
            throw InvalidInput("a block index is out of range");
        }
    }
    for (const std::int64_t id : strand_id) {
        if (id < 0 || id >= size_) {
            throw InvalidInput("a strand index is outside the vector");
        }
    }
    for (const std::int32_t t : type) {
        if (t < 0 || t >= types) {
            throw InvalidInput("a branch type is out of range");
        }
    }
    for (const std::int32_t s : sources) {
        if (s < 0 || s >= count) {
            throw InvalidInput("a source is not a branch of the family");
        }
    }
    Family family;
    family.planes = planes;
    family.rows = rows;
    family.cols = cols;
    family.lookup = std::move(lookup);
    family.cells = std::move(cells);
    family.type = std::move(type);
    family.strand_start = std::move(strand_start);
    family.strand_id = std::move(strand_id);
    family.sources = std::move(sources);
    family.slot_row = std::move(slot_row);
    family.slot_col = std::move(slot_col);
    family.partner_start = std::move(partner_start);
    family.partner = std::move(partner);
    family.types = types;
    family.block = std::move(block);
    family.width = width;
    family.correction = std::move(correction);
    family.exact = std::move(exact);
    const auto source_count = static_cast<std::int64_t>(family.sources.size());
    family.mirror_start.assign(static_cast<size_t>(source_count) + 1U, 0);
    for (std::int64_t i = 0; i < source_count; ++i) {
        std::int64_t count = 0;
        walk(family, i, [&](size_t, std::int64_t, std::int64_t, const std::int64_t t_first, std::int64_t,
                            const bool uncut) {
            if (uncut) {
                family.mirror_target.push_back(family.strand_id[static_cast<size_t>(t_first)]);
                ++count;
            }
        });
        family.mirror_start[static_cast<size_t>(i) + 1U] = family.mirror_start[static_cast<size_t>(i)] + count;
    }
    families_.push_back(std::move(family));
}

// Calls visit(block_offset, source strands (first, count), target strands (first, count), target_uncut)
// for every neighbour of one source branch that has a block.
template <typename Visit>
void StrandCorrection::walk(const Family& f, const std::int64_t source_index, Visit&& visit) const {
    const std::int32_t s = f.sources[static_cast<size_t>(source_index)];
    const std::int32_t plane = f.cells[3 * static_cast<size_t>(s)];
    const std::int32_t row = f.cells[3 * static_cast<size_t>(s) + 1];
    const std::int32_t col = f.cells[3 * static_cast<size_t>(s) + 2];
    const std::int32_t ts = f.type[static_cast<size_t>(s)];
    const std::int64_t s_first = f.strand_start[static_cast<size_t>(s)];
    const std::int64_t s_count = f.strand_start[static_cast<size_t>(s) + 1] - s_first;
    const auto slots = static_cast<std::int64_t>(f.slot_row.size());
    const std::int64_t types = f.types;
    for (std::int64_t k = 0; k < slots; ++k) {
        const std::int64_t r = row + f.slot_row[static_cast<size_t>(k)];
        const std::int64_t c = col + f.slot_col[static_cast<size_t>(k)];
        if (r < 0 || r >= f.rows || c < 0 || c >= f.cols) {
            continue;
        }
        for (std::int32_t p = f.partner_start[static_cast<size_t>(plane)];
             p < f.partner_start[static_cast<size_t>(plane) + 1]; ++p) {
            const std::int64_t other = f.partner[static_cast<size_t>(p)];
            const std::int32_t t = f.lookup[static_cast<size_t>((other * f.rows + r) * f.cols + c)];
            if (t < 0) {
                continue;
            }
            const std::int32_t tt = f.type[static_cast<size_t>(t)];
            const std::int32_t b =
                f.block[static_cast<size_t>((((k * f.planes + plane) * f.planes + other) * types + ts) * types + tt)];
            if (b < 0) {
                continue;
            }
            const std::int64_t t_first = f.strand_start[static_cast<size_t>(t)];
            const std::int64_t t_count = f.strand_start[static_cast<size_t>(t) + 1] - t_first;
            visit(static_cast<size_t>(b) * static_cast<size_t>(f.width * f.width), s_first, s_count, t_first, t_count,
                  tt == 0);
        }
    }
}

void StrandCorrection::apply(const double* const x, double* const y, const int threads) const {
    std::fill(y, y + size_, 0.0);
    for (const Family& f : families_) {
        const auto sources = static_cast<std::int64_t>(f.sources.size());
        const std::int64_t w = f.width;
        const double* const blocks = f.correction.data();
        std::vector<double> mirrored(f.mirror_target.size(), 0.0);
        // A source writes only its own strands and its own run of mirrored
        // entries; the mirrored entries are added to their uncut neighbours
        // afterwards, in source order.  Every sum runs in a fixed order, so the
        // result is the same at any thread count.
#pragma omp parallel for schedule(dynamic, 64) num_threads(std::max(threads, 1)) if (threads > 1)
        for (std::int64_t i = 0; i < sources; ++i) {
            std::int64_t next = f.mirror_start[static_cast<size_t>(i)];
            walk(f, i, [&](const size_t offset, const std::int64_t s_first, const std::int64_t s_count,
                           const std::int64_t t_first, const std::int64_t t_count, const bool uncut) {
                const double* const block = blocks + offset;
                for (std::int64_t a = 0; a < s_count; ++a) {
                    double sum = 0.0;
                    for (std::int64_t b = 0; b < t_count; ++b) {
                        sum += block[a * w + b] * x[f.strand_id[static_cast<size_t>(t_first + b)]];
                    }
                    y[f.strand_id[static_cast<size_t>(s_first + a)]] += sum;
                }
                if (uncut) {
                    double sum = 0.0;
                    for (std::int64_t a = 0; a < s_count; ++a) {
                        sum += block[a * w] * x[f.strand_id[static_cast<size_t>(s_first + a)]];
                    }
                    mirrored[static_cast<size_t>(next++)] = sum;
                }
            });
        }
        for (size_t k = 0; k < mirrored.size(); ++k) {
            y[f.mirror_target[k]] += mirrored[k];
        }
    }
}

void StrandCorrection::entries(const bool exact, std::vector<std::int64_t>& row, std::vector<std::int64_t>& col,
                               std::vector<double>& value) const {
    row.clear();
    col.clear();
    value.clear();
    for (const Family& f : families_) {
        const std::int64_t w = f.width;
        const double* const blocks = exact ? f.exact.data() : f.correction.data();
        const auto sources = static_cast<std::int64_t>(f.sources.size());
        for (std::int64_t i = 0; i < sources; ++i) {
            walk(f, i, [&](const size_t offset, const std::int64_t s_first, const std::int64_t s_count,
                           const std::int64_t t_first, const std::int64_t t_count, const bool uncut) {
                const double* const block = blocks + offset;
                for (std::int64_t a = 0; a < s_count; ++a) {
                    for (std::int64_t b = 0; b < t_count; ++b) {
                        const std::int64_t si = f.strand_id[static_cast<size_t>(s_first + a)];
                        const std::int64_t tj = f.strand_id[static_cast<size_t>(t_first + b)];
                        row.push_back(si);
                        col.push_back(tj);
                        value.push_back(block[a * w + b]);
                        if (uncut) {
                            row.push_back(tj);
                            col.push_back(si);
                            value.push_back(block[a * w + b]);
                        }
                    }
                }
            });
        }
    }
}

std::int64_t StrandCorrection::stored_bytes() const noexcept {
    std::int64_t total = 0;
    for (const Family& f : families_) {
        total += static_cast<std::int64_t>(
            sizeof(std::int32_t) * (f.lookup.size() + f.cells.size() + f.type.size() + f.sources.size() +
                                    f.slot_row.size() + f.slot_col.size() + f.partner_start.size() +
                                    f.partner.size() + f.block.size()) +
            sizeof(std::int64_t) * (f.strand_start.size() + f.strand_id.size() + f.mirror_start.size() +
                                    f.mirror_target.size()) +
            sizeof(double) * (f.correction.size() + f.exact.size()));
    }
    return total;
}

}  // namespace pcbcore::sheet
