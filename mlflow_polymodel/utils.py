"""..."""

from collections.abc import Callable
from typing import Any

from mlflow_polymodel.log import LogModelFunction


def wrap_log(
    model_keyword: str,
    log_function: Callable[..., Any],
) -> LogModelFunction:
    """..."""
    return lambda model, *args, **kwargs: log_function(
        *args,
        **{model_keyword: model},
        **kwargs,
    )
