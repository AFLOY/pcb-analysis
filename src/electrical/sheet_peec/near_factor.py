"""Factorisations of the near-field saddle-point preconditioner shared by the CPU and CUDA paths.

The ``near`` preconditioner solves ``[[Z, -A], [A^T, 0]]`` with ``Z`` the
near branch impedance (complex symmetric) and ``A`` the reduced incidence.
SciPy's SuperLU factors it as a general matrix.  The form here negates the
node rows and regularizes the zero block,

    [[Z, -A], [-A^T, -delta I]],

which is complex symmetric and quasi-definite, so a symmetric LDL^T applies:
cuDSS on the device (:mod:`.sheet_cuda`) and MUMPS on the CPU
(:class:`MumpsQuasiDefiniteNear`, in a core built with
``-DPCB_NATIVE_MUMPS=ON``).

On the strip pair at 0.2 mm with seven skin filaments a layer (47k branches,
30 MHz, one thread) SuperLU took 60 s to factor and 0.18 s a solve with
159M factor entries; MUMPS 5.9.1 in METIS order (AOCC, single-threaded
OpenBLAS) took 3.5 s, 0.07 s a solve and 39M entries, both solving the
saddle to 3e-10.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import scipy.sparse as sp

from electrical import backend as _backend
from electrical.threads import thread_budget

NEAR_FACTORIZATIONS_CPU = ("auto", "superlu", "mumps_quasi_definite")

# MUMPS's CNTL(1).  Without threshold pivoting (0) the quasi-definite factors
# solved the strip pair's near matrix only to 1e-2; 0.01 gave 4e-10 with 13 %
# more factor entries.
MUMPS_PIVOT_THRESHOLD = 0.01

# The largest relative residual of the check solve a new factor must pass.
# It catches factors that are garbage (1e25 to NaN from a BLAS that was not
# thread-safe), not ill-conditioned ones: a thick inlay cut into many
# filaments, whose mutual terms nearly equal their self terms, solved its own
# near matrix to 3e-5, which GMRES corrects as with any preconditioner.
MUMPS_CHECK_RESIDUAL = 1e-3

# MUMPS ICNTL settings.  ICNTL(12) = 1, the usual symmetric ordering: MUMPS
# 5.9.1's automatic choice took the compressed ordering meant for augmented
# systems, and on the strip pair at 0.2 mm (47k branches) that gave 23.5k
# delayed pivots, 72M factor entries and a 30.6 s factorisation; the usual
# ordering gave 14.8k, 39M and 3.5 s (one thread, METIS, AOCC build).
#
# ICNTL(48) = 0: no L0 tree parallelism.  With it MUMPS assigned subtrees to
# threads dynamically and the factors' last bits changed from run to run at a
# fixed thread count (three runs at 8 threads, three digests); without it the
# solves gave the same bits in repeated runs and at 8 and 48 threads, at
# 0.90 to 1.32 of the time (8 threads; LM5145EVM-2L real-fill SW conductor
# and the hot loop at 0.2 and 0.1 mm, EPYC 9965).
MUMPS_CONTROLS = {12: 1, 48: 0}

# MUMPS's own threads, by default.  Without its tree parallelism the factors
# are the same bits run after run at one thread count, but not across counts
# (one and two, three, and four and more gave three digests on a small mesh),
# so the factor runs on one thread unless the caller asks for more
# (``near_factor_threads``): every thread budget then gives the same bits.
# More threads are faster (8 against 1: 2.1x on the real-fill SW conductor,
# 1.4x on the hot loop at 0.2 mm, EPYC 9965) and give the same bits only at
# the same count.
MUMPS_DEFAULT_THREADS = 1


def resolve_near_factor_threads(value: Any) -> int:
    """``near_factor_threads``: an integer, or ``"budget"`` for the process's thread budget."""
    if value is None:
        return MUMPS_DEFAULT_THREADS
    if value == "budget":
        return thread_budget()
    count = int(value)
    if count < 1:
        raise ValueError("near_factor_threads must be at least 1 or 'budget'")
    return count

# The regularization ``delta`` (over the median |Z_ii|) by factorization.
# MUMPS pivots, so it needs none to be stable; a large one only weakens the
# preconditioner: at 1e-8 (cuDSS's, which does not pivot) the strip pair at
# 0.2 mm and 30 MHz took 384 GMRES iterations, at 1e-12 306 and at 1e-16 299
# (SuperLU's count).
NEAR_REGULARIZATION = {"superlu": 0.0, "mumps_quasi_definite": 1e-12}


def resolve_near_factorization(name: str) -> str:
    """``auto``: MUMPS where the C++ core links it, SuperLU otherwise."""
    if name not in NEAR_FACTORIZATIONS_CPU:
        raise ValueError(f"near_factorization must be one of {NEAR_FACTORIZATIONS_CPU}")
    if name == "auto":
        return "mumps_quasi_definite" if mumps_available() else "superlu"
    return name


def quasi_definite_near_matrix(near: Any, reduced: Any, regularization: float) -> tuple[Any, np.ndarray, float]:
    """``(matrix, sign, delta)``: the near saddle-point matrix made quasi-definite.

    ``matrix`` is ``[[Z, -A], [-A^T, -delta I]]`` (SciPy CSR, complex128,
    sorted), ``sign`` the row signs that take the saddle-point right-hand side
    to it (``+1`` on branches, ``-1`` on nodes), ``delta`` the regularization
    over the median ``|Z_ii|``.
    """
    near = sp.csr_matrix(near)
    diagonal = np.abs(near.diagonal())
    scale = float(np.median(diagonal[diagonal > 0.0])) if np.any(diagonal > 0.0) else 1.0
    delta = float(regularization) / scale
    nodes = reduced.shape[1]
    matrix = sp.bmat(
        [[near, -reduced], [-reduced.T, -delta * sp.identity(nodes, format="csr")]], format="csr"
    ).astype(np.complex128)
    matrix.sum_duplicates()
    matrix.sort_indices()
    sign = np.concatenate([np.ones(near.shape[0]), -np.ones(nodes)]).astype(np.complex128)
    return matrix, sign, delta


def mumps_available() -> bool:
    """True when the C++ core is built and linked against MUMPS."""
    core = _backend.core()
    linalg = getattr(core, "linalg", None) if core is not None else None
    return linalg is not None and getattr(linalg, "mumps_version", None) is not None


# Analyses kept for a mesh's next frequency: the near pattern does not depend
# on the frequency, and at 0.1 mm the METIS analysis was half the time of an
# eight-thread factorisation.  A factor is handed to one preconditioner at a
# time and comes back here when that solve closes it; one is kept.
_ANALYSES: dict[bytes, Any] = {}


def _pattern_key(upper: sp.csr_matrix) -> bytes:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(upper.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(upper.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(upper.indices, dtype=np.int64).tobytes())
    return digest.digest()


def clear_near_factor_cache() -> None:
    """Drop the kept analyses (and the factors they hold)."""
    _ANALYSES.clear()


class MumpsQuasiDefiniteNear:
    """The near saddle-point preconditioner as a MUMPS LDL^T of its quasi-definite form.

    ``solve(v)`` returns ``M^-1 v`` for the unregularized sign convention,
    for a vector or an ``(n, k)`` block (all columns in one MUMPS solve).
    ``close()`` hands the analysis back for the next frequency of the mesh.
    """

    def __init__(
        self, near: Any, reduced: Any, regularization: float = NEAR_REGULARIZATION["mumps_quasi_definite"],
        threads: int | None = None,
    ) -> None:
        if not mumps_available():
            raise ValueError(
                "near_factorization='mumps_quasi_definite' needs the C++ core built with -DPCB_NATIVE_MUMPS=ON"
            )
        core = _backend.core()
        self.threads = MUMPS_DEFAULT_THREADS if threads is None else int(threads)
        matrix, sign, delta = quasi_definite_near_matrix(near, reduced, regularization)
        upper = sp.triu(matrix, format="csr")
        upper.sort_indices()
        self.size = int(upper.shape[0])
        self.delta = delta
        self._sign = sign.real.copy()
        self._key = _pattern_key(upper) + self.threads.to_bytes(4, "little")
        ldlt = _ANALYSES.pop(self._key, None)
        self.reused_analysis = ldlt is not None
        if ldlt is None:
            ldlt = core.linalg.SymmetricLdlt(
                self.size,
                np.ascontiguousarray(upper.indptr, dtype=np.int64),
                np.ascontiguousarray(upper.indices, dtype=np.int64),
                ordering="metis",
                threads=self.threads,
                controls=MUMPS_CONTROLS,
            )
        ldlt.factor(np.ascontiguousarray(upper.data, dtype=np.complex128), MUMPS_PIVOT_THRESHOLD, self.threads)
        self._ldlt = ldlt
        # One check solve: a factor that does not solve its own matrix is
        # refused, not handed to GMRES.  (A MUMPS linked against an OpenBLAS
        # built without USE_LOCKING returned factors with residuals of 1e25 to
        # NaN on 8-48 threads, and GMRES then diverged without an error.)
        probe = np.random.default_rng(0).standard_normal(self.size).astype(np.complex128)
        check = ldlt.solve(probe, self.threads)
        self.check_residual = float(np.linalg.norm(matrix @ check - probe) / np.linalg.norm(probe))
        if not self.check_residual <= MUMPS_CHECK_RESIDUAL:
            _ANALYSES.pop(self._key, None)
            raise RuntimeError(
                f"the MUMPS near factor solves its own matrix only to {self.check_residual:.1e} "
                f"(threads={self.threads}); is its BLAS thread-safe (OpenBLAS USE_LOCKING=1)?"
            )
        self.analysis_seconds = 0.0 if self.reused_analysis else float(ldlt.analysis_seconds)
        self.factor_seconds = float(ldlt.factor_seconds)
        self.factor_entries = int(ldlt.factor_entries)

    def solve(self, vector: np.ndarray) -> np.ndarray:
        if self._ldlt is None:
            raise RuntimeError("the near factor was closed")
        sign = self._sign if vector.ndim == 1 else self._sign[:, None]
        return self._ldlt.solve(np.asarray(vector, dtype=np.complex128) * sign, self.threads)

    __call__ = solve

    def close(self) -> None:
        """Keep the analysis for this pattern's next factorisation and release this one."""
        if self._ldlt is None:
            return
        _ANALYSES.clear()
        _ANALYSES[self._key] = self._ldlt
        self._ldlt = None
