"""DICE-PEEC: delta scoring of layout candidates on a fixed 2.5D grid.

``multilayer_peec`` holds the scalar interaction proxy that ranks many
candidate shapes cheaply against an accepted layout; ``layout_ops`` compiles a
router's segment and via edits into that proxy's sparse deltas;
``controller`` and ``backends`` decide where and how a scoring plan runs and
record what it cost; ``stackup`` names the layers and their heights.  It does
not solve for current or potential: the sheet solve is
``electrical.sheet_peec`` and the 3D voxel solve ``electrical.voxel_peec``.
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
