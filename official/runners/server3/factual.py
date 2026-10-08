"""Load the shared factual evaluator owned by server1.

The server3 runner must use ``official.evaluation.factual`` for both
CounterFact and zsRE.  No task-local metric implementation belongs here.
"""

from importlib import import_module
from types import ModuleType


class SharedFactualEvaluatorUnavailable(RuntimeError):
    """The common factual evaluator has not been published yet."""


def require_shared_factual() -> ModuleType:
    """Return the common evaluator or fail before any scientific evaluation."""
    module_name = "official.evaluation.factual"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise SharedFactualEvaluatorUnavailable(
                "SHARED_FACTUAL_EVALUATOR_NOT_PUBLISHED: " + module_name
            ) from error
        raise
