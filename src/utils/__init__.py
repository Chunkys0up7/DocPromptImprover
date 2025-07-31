"""
Utility functions for the evaluation-only framework.

This module contains configuration management, logging setup, and other
utility functions used throughout the statistical evaluation system.
"""

from .config import (
    get_config,
    get_settings,
    validate_config,
    update_config,
    is_development_mode,
    get_data_directory,
    get_results_directory,
    get_temp_directory,
    get_log_directory
)

from .logging import (
    get_logger,
    setup_logging,
    DocumentProcessingLogger,
    MetricsLogger
)

__all__ = [
    # Configuration
    "get_config",
    "get_settings", 
    "validate_config",
    "update_config",
    "is_development_mode",
    "get_data_directory",
    "get_results_directory",
    "get_temp_directory",
    "get_log_directory",
    
    # Logging
    "get_logger",
    "setup_logging",
    "DocumentProcessingLogger",
    "MetricsLogger"
] 