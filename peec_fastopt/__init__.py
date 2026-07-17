"""Optimization-oriented PEEC acceleration for single- and multilayer PCB work.

Production CUDA path (``cuda_pypeec``) accelerates PyPEEC solves supplied by
plane_opt.  Multilayer DICE scaffolding (stackup, 2.5D operator, layout ops)
prepares sparse candidate scoring while plane_opt adds multilayer routing.
"""

from .layout_ops import (
    CandidateEdit,
    CompiledCandidate,
    SegmentOp,
    ViaOp,
    compile_candidate,
    compile_many,
)
from .multilayer_peec import (
    FFTInteraction25D,
    MultilayerDeltaScorer,
    SparseDeltaML,
    ViaSet,
    ViaSpec,
)
from .stackup import Stackup

__all__ = [
    "CandidateEdit",
    "CompiledCandidate",
    "FFTInteraction25D",
    "MultilayerDeltaScorer",
    "SegmentOp",
    "SparseDeltaML",
    "Stackup",
    "ViaOp",
    "ViaSet",
    "ViaSpec",
    "compile_candidate",
    "compile_many",
]
