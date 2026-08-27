"""Typed scientific boundaries for BGODE-R1."""


class BGODEError(RuntimeError):
    """Base class for a violated BGODE implementation contract."""


class BGODEScientificBoundary(BGODEError):
    """The requested operation is outside the locked scientific method."""


class UnsupportedBatchBoundary(BGODEScientificBoundary):
    """BGODE-R1 supports exactly one request per atomic edit."""


class EventPartitionBoundary(BGODEScientificBoundary):
    """The source/target event partition is not semantically well-defined."""


class NumericalRankBoundary(BGODEScientificBoundary):
    """The equality Rayleighian lacks the required numerical rank/range."""


class NativeBypassBoundary(BGODEScientificBoundary):
    """The explicit native bypass was requested outside its exact special case."""
