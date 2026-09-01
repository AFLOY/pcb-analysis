"""Low-precision vector runtimes used by matrix-free MPIR.

The outer iterative-refinement loop deliberately remains in NumPy/FP64.  Only
this small runtime contract and an operator's low-precision ``apply`` need a
device implementation.  That is the intended migration boundary for CUDA and
dataflow accelerators.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


RuntimeBackend = Literal["cpu", "cuda", "auto"]


@runtime_checkable
class LowPrecisionRuntime(Protocol):
    """Primitive operations required by the inner PCG solve."""

    name: str
    namespace: Any
    dtype: Any
    is_cuda: bool

    def from_host(self, value: np.ndarray) -> Any: ...

    def to_host(self, value: Any) -> np.ndarray: ...

    def zeros_like(self, value: Any) -> Any: ...

    def copy(self, value: Any) -> Any: ...

    def dot(self, left: Any, right: Any) -> complex | float: ...

    def norm(self, value: Any) -> float: ...

    def axpy(self, alpha: complex | float, x: Any, y: Any) -> Any: ...

    def divide(self, numerator: Any, denominator: Any) -> Any: ...

    def synchronize(self) -> None: ...


class NumpyFloat32Runtime:
    """Reference FP32 runtime; also useful when no accelerator is installed."""

    name = "numpy-fp32"
    namespace = np
    dtype = np.float32
    is_cuda = False

    def from_host(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=self.dtype)

    def to_host(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    def zeros_like(self, value: np.ndarray) -> np.ndarray:
        return np.zeros_like(value, dtype=self.dtype)

    def copy(self, value: np.ndarray) -> np.ndarray:
        return np.array(value, dtype=self.dtype, copy=True)

    def dot(self, left: np.ndarray, right: np.ndarray) -> float:
        # NumPy accumulates a float32 dot in float32.  Conversion happens only
        # after the intentionally low-precision reduction has completed.
        return float(np.vdot(left, right).real)

    def norm(self, value: np.ndarray) -> float:
        return float(np.linalg.norm(value))

    def axpy(
        self,
        alpha: float,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        scalar = np.asarray(alpha, dtype=self.dtype)
        return np.asarray(y + scalar * x, dtype=self.dtype)

    def divide(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(numerator / denominator, dtype=self.dtype)

    def synchronize(self) -> None:
        return None


def _initialise_cupy_runtime(runtime: Any, device_id: int) -> None:
    """Attach a CuPy runtime to one visible CUDA device.

    Importing CuPy alone does not prove that a driver and a device are usable.
    Failing here gives callers a useful error before FEM coefficients are
    partially allocated.
    """

    try:
        device_count = int(runtime.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on local CUDA driver
        raise RuntimeError(
            "CUDA backend requested, but the NVIDIA driver reported no usable device"
        ) from exc
    if device_count < 1:
        raise RuntimeError(
            "CUDA backend requested, but the NVIDIA driver reported no usable device"
        )
    if device_id < 0 or device_id >= device_count:
        raise ValueError(
            f"CUDA device_id {device_id} is outside the visible range "
            f"[0, {device_count - 1}]"
        )
    runtime.cuda.Device(device_id).use()


class CupyFloat32Runtime:
    """Optional CUDA FP32 runtime.

    CuPy is imported only when this runtime is constructed, so CPU installs do
    not acquire a CUDA dependency.  FP64 residuals and updates stay on the
    host; one correction vector crosses the device boundary per outer step.
    """

    name = "cupy-fp32"
    is_cuda = True

    def __init__(self, *, device_id: int = 0) -> None:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional
            raise RuntimeError(
                "CupyFloat32Runtime requires the 'cuda' optional dependency"
            ) from exc
        self.namespace = cp
        self.dtype = cp.float32
        self.device_id = int(device_id)
        _initialise_cupy_runtime(cp, self.device_id)

    def from_host(self, value: np.ndarray) -> Any:
        return self.namespace.asarray(value, dtype=self.dtype)

    def to_host(self, value: Any) -> np.ndarray:
        return np.asarray(self.namespace.asnumpy(value), dtype=np.float64)

    def zeros_like(self, value: Any) -> Any:
        return self.namespace.zeros_like(value, dtype=self.dtype)

    def copy(self, value: Any) -> Any:
        return self.namespace.array(value, dtype=self.dtype, copy=True)

    def dot(self, left: Any, right: Any) -> float:
        return float(self.namespace.vdot(left, right).real.item())

    def norm(self, value: Any) -> float:
        return float(self.namespace.linalg.norm(value).item())

    def axpy(self, alpha: float, x: Any, y: Any) -> Any:
        scalar = self.namespace.asarray(alpha, dtype=self.dtype)
        return self.namespace.asarray(y + scalar * x, dtype=self.dtype)

    def divide(self, numerator: Any, denominator: Any) -> Any:
        return self.namespace.asarray(numerator / denominator, dtype=self.dtype)

    def synchronize(self) -> None:
        self.namespace.cuda.get_current_stream().synchronize()


class NumpyComplex64Runtime:
    """Reference complex64 runtime for frequency-domain inner solves."""

    name = "numpy-complex64"
    namespace = np
    dtype = np.complex64
    is_cuda = False

    def from_host(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=self.dtype)

    def to_host(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.complex128)

    def zeros_like(self, value: np.ndarray) -> np.ndarray:
        return np.zeros_like(value, dtype=self.dtype)

    def copy(self, value: np.ndarray) -> np.ndarray:
        return np.array(value, dtype=self.dtype, copy=True)

    def dot(self, left: np.ndarray, right: np.ndarray) -> complex:
        return complex(np.vdot(left, right))

    def norm(self, value: np.ndarray) -> float:
        return float(np.linalg.norm(value))

    def axpy(
        self,
        alpha: complex | float,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        scalar = np.asarray(alpha, dtype=self.dtype)
        return np.asarray(y + scalar * x, dtype=self.dtype)

    def divide(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(numerator / denominator, dtype=self.dtype)

    def synchronize(self) -> None:
        return None


class CupyComplex64Runtime:
    """Optional CUDA complex64 runtime for frequency-domain inner solves."""

    name = "cupy-complex64"
    is_cuda = True

    def __init__(self, *, device_id: int = 0) -> None:
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional
            raise RuntimeError(
                "CupyComplex64Runtime requires the 'cuda' optional dependency"
            ) from exc
        self.namespace = cp
        self.dtype = cp.complex64
        self.device_id = int(device_id)
        _initialise_cupy_runtime(cp, self.device_id)

    def from_host(self, value: np.ndarray) -> Any:
        return self.namespace.asarray(value, dtype=self.dtype)

    def to_host(self, value: Any) -> np.ndarray:
        return np.asarray(self.namespace.asnumpy(value), dtype=np.complex128)

    def zeros_like(self, value: Any) -> Any:
        return self.namespace.zeros_like(value, dtype=self.dtype)

    def copy(self, value: Any) -> Any:
        return self.namespace.array(value, dtype=self.dtype, copy=True)

    def dot(self, left: Any, right: Any) -> complex:
        return complex(self.namespace.vdot(left, right).item())

    def norm(self, value: Any) -> float:
        return float(self.namespace.linalg.norm(value).item())

    def axpy(self, alpha: complex | float, x: Any, y: Any) -> Any:
        scalar = self.namespace.asarray(alpha, dtype=self.dtype)
        return self.namespace.asarray(y + scalar * x, dtype=self.dtype)

    def divide(self, numerator: Any, denominator: Any) -> Any:
        return self.namespace.asarray(numerator / denominator, dtype=self.dtype)

    def synchronize(self) -> None:
        self.namespace.cuda.get_current_stream().synchronize()


def cuda_available() -> bool:
    """Return whether CuPy can see at least one usable CUDA device."""

    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # pragma: no cover - result depends on local installation
        return False


def make_float32_runtime(
    backend: RuntimeBackend = "cpu",
    *,
    device_id: int = 0,
) -> LowPrecisionRuntime:
    """Construct the real low-precision runtime requested by a front end."""

    if backend == "cpu":
        return NumpyFloat32Runtime()
    if backend == "cuda":
        return CupyFloat32Runtime(device_id=device_id)
    if backend == "auto":
        return (
            CupyFloat32Runtime(device_id=device_id)
            if cuda_available()
            else NumpyFloat32Runtime()
        )
    raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")


def make_complex64_runtime(
    backend: RuntimeBackend = "cpu",
    *,
    device_id: int = 0,
) -> LowPrecisionRuntime:
    """Construct the complex low-precision runtime requested by a front end."""

    if backend == "cpu":
        return NumpyComplex64Runtime()
    if backend == "cuda":
        return CupyComplex64Runtime(device_id=device_id)
    if backend == "auto":
        return (
            CupyComplex64Runtime(device_id=device_id)
            if cuda_available()
            else NumpyComplex64Runtime()
        )
    raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
