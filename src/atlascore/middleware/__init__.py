from ._approval import ApprovalMiddleware
from ._base import BaseMiddleware, MiddlewareContext
from ._chain import MiddlewareChain
from ._logging import LoggingMiddleware

__all__ = [
    "BaseMiddleware",
    "MiddlewareContext",
    "MiddlewareChain",
    "LoggingMiddleware",
    "ApprovalMiddleware",
]

try:
    from ._otel import (  # noqa: F401
        OTelMiddleware,
        auto_instrument,
    )

    __all__.extend(["OTelMiddleware", "auto_instrument"])
except ImportError:
    pass
