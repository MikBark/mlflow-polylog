"""Top-level module for the mlflow_polymodel package.

This module exposes the main imperative logging interface for the polymodel library,
including functions for logging models and registering custom log handlers.
"""

from mlflow_polymodel.defaults import get_default_log
from mlflow_polymodel.functions import log_model, register_log
from mlflow_polymodel.log import PolymorphicModelLog
