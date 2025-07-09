"""This module provides a polymorphic model logging utility for MLflow.

The module defines the PolymorphicModelLog class, which enables logging of models to
MLflow without requiring knowledge of the model's type or the installation of all possible
machine learning libraries. It leverages a type-based mapping to associate model types
with their corresponding logging functions.
"""

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from mlflow_polymodel.type_mapping import TypeMapping

LogModelParamSpec = ParamSpec('LogModelParamSpec')
ModelType = TypeVar('ModelType')
LogModelFunctionType = Callable[
    [ModelType, LogModelParamSpec.args, LogModelParamSpec.kwargs],
    None,
]


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


class PolymorphicModelLog:
    """Log models to MLflow in a type-agnostic and extensible manner.

    This class allows users to log models to MLflow without knowing the specific model
    type or installing all available ML libraries. It maintains a mapping between model
    types and their respective logging functions, enabling flexible and extensible model
    logging.
    """

    def __init__(
        self,
        log_map: TypeMapping[LogModelFunctionType],
    ) -> None:
        """Initialize the PolymorphicModelLog with a type-to-logging-function mapping.

        Args:
            log_map : A TypeMapping that associates model types with their corresponding
                logging functions. The logging functions must accept the model and any
                additional arguments required for logging.
        """
        self._log_map = log_map

    def __call__(
        self,
        model: Any,
        *args: LogModelParamSpec.args,
        **kwargs: LogModelParamSpec.kwargs,
    ) -> None:
        """Log the given model using the appropriate logging function.

        Determines the correct logging function for the provided model based on its type
        and invokes it with the supplied arguments.

        Args:
            model : The model instance to be logged. Its type is used to select the
                appropriate logging function.
            *args : Positional arguments to pass to the logging function.
            **kwargs : Keyword arguments to pass to the logging function.

        Returns:
            None. The function performs logging as a side effect.

        Raises:
            KeyError : If no logging function is registered for the model's type.
            Exception : Propagates any exception raised by the underlying
                logging function.
        """
        log_function: LogModelFunctionType = self._log_map[model]
        log_function(model, *args, **kwargs)

    def add_log(
        self,
        model_type: type,
        log_model_function: Callable[..., None],
    ) -> 'PolymorphicModelLog':
        """Return a new PolymorphicModelLog with an additional model logging function.

        Creates and returns a new PolymorphicModelLog instance with the provided model
        type and its corresponding logging function added to the mapping.

        Args:
            model_type : The type of model to associate with the logging function.
            log_model_function : The function to use for logging models of the specified
                type. Must accept the model instance and any required arguments.

        Returns:
            A new PolymorphicModelLog instance with the updated type-to-function mapping.

        Raises:
            TypeError : If model_type is not a valid type or log_model_function is not
                callable.
        """
        return PolymorphicModelLog(
            TypeMapping(self._log_map, {model_type: log_model_function}),
        )
