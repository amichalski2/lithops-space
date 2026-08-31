"""Errors shared by domain ports and their adapters."""


class LithopsError(RuntimeError):
    """Base error for expected Lithops failures."""


class RepositoryError(LithopsError):
    """Base persistence error."""


class NotFoundError(RepositoryError):
    """Requested record does not exist."""


class ConflictError(RepositoryError):
    """Write conflicts with an existing or newer record."""


class OperationInProgressError(ConflictError):
    """An operation with the same request key is already running."""


class BenchmarkError(LithopsError):
    """Base exception for benchmark integration failures."""


class BenchmarkContractError(BenchmarkError):
    """A request violates the public benchmark contract."""
