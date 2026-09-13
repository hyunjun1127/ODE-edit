"""FzCB Hard-Alpha with Official dense ``cache_c``.

The package is intentionally separate from the server4 FzCB prototype.  It
contains the dense Alpha backend, matrix-free completion controller, cache
transaction, and thin experiment adapters used by the server2 HA0--HA2 run.
"""

from .contracts import (
    EngineeringBoundary,
    MethodBoundary,
    NumericalLock,
    TrajectoryBoundary,
)

__all__ = [
    "EngineeringBoundary",
    "MethodBoundary",
    "NumericalLock",
    "TrajectoryBoundary",
]
