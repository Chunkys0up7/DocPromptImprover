"""
Utility functions for the DSPy-Pydantic document processing system.

This module contains configuration management, logging setup, and other
utility functions used throughout the system.
"""

from .config import (
    get_config,
    get_settings,
    validate_config,
    update_config,
    get_api_key,
    is_development_mode,
    get_allowed_extensions,
    get_optimization_config,
    get_quality_control_config
)

from .logging import (
    get_logger,
    setup_logging,
    DocumentProcessingLogger,
    MetricsLogger,
    get_processing_logger,
    get_metrics_logger,
    create_processing_logger,
    create_metrics_logger
)

__all__ = [
    # Configuration
    "get_config",
    "get_settings", 
    "validate_config",
    "update_config",
    "get_api_key",
    "is_development_mode",
    "get_allowed_extensions",
    "get_optimization_config",
    "get_quality_control_config",
    
    # Logging
    "get_logger",
    "setup_logging",
    "DocumentProcessingLogger",
    "MetricsLogger", 
    "get_processing_logger",
    "get_metrics_logger",
    "create_processing_logger",
    "create_metrics_logger"
] 