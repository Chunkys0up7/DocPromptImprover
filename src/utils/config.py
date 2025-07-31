"""
Configuration management for the evaluation-only framework.

This module handles loading and managing configuration from environment
variables and configuration files for the statistical evaluation service.
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Evaluation Configuration
    default_confidence_threshold: float = Field(0.7, env="DEFAULT_CONFIDENCE_THRESHOLD")
    evaluation_timeout: int = Field(30, env="EVALUATION_TIMEOUT")
    max_concurrent_evaluations: int = Field(10, env="MAX_CONCURRENT_EVALUATIONS")
    
    # Statistical Analysis Configuration
    min_sample_size: int = Field(10, env="MIN_SAMPLE_SIZE")
    confidence_interval: float = Field(0.95, env="CONFIDENCE_INTERVAL")
    statistical_significance_threshold: float = Field(0.05, env="STATISTICAL_SIGNIFICANCE_THRESHOLD")
    
    # Pattern Detection Configuration
    pattern_detection_enabled: bool = Field(True, env="PATTERN_DETECTION_ENABLED")
    min_pattern_frequency: int = Field(3, env="MIN_PATTERN_FREQUENCY")
    pattern_confidence_threshold: float = Field(0.8, env="PATTERN_CONFIDENCE_THRESHOLD")
    
    # Performance Monitoring
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    performance_tracking_enabled: bool = Field(True, env="PERFORMANCE_TRACKING_ENABLED")
    trend_analysis_enabled: bool = Field(True, env="TREND_ANALYSIS_ENABLED")
    
    # Database Configuration (for storing evaluation results)
    database_url: str = Field("sqlite:///./evaluation_results.db", env="DATABASE_URL")
    results_retention_days: int = Field(90, env="RESULTS_RETENTION_DAYS")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    cors_origins: str = Field("*", env="CORS_ORIGINS")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file: str = Field("logs/evaluation_service.log", env="LOG_FILE")
    
    # File Storage
    data_dir: str = Field("data", env="DATA_DIR")
    results_dir: str = Field("results", env="RESULTS_DIR")
    temp_dir: str = Field("temp", env="TEMP_DIR")
    max_file_size: int = Field(10485760, env="MAX_FILE_SIZE")  # 10MB
    
    # Security
    secret_key: str = Field("your_secret_key_here", env="SECRET_KEY")
    enable_cors: bool = Field(True, env="ENABLE_CORS")
    
    # Development
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_config() -> Dict[str, Any]:
    """
    Get configuration as a dictionary.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    settings = get_settings()
    return {
        "evaluation": {
            "default_confidence_threshold": settings.default_confidence_threshold,
            "evaluation_timeout": settings.evaluation_timeout,
            "max_concurrent_evaluations": settings.max_concurrent_evaluations,
        },
        "statistical_analysis": {
            "min_sample_size": settings.min_sample_size,
            "confidence_interval": settings.confidence_interval,
            "statistical_significance_threshold": settings.statistical_significance_threshold,
        },
        "pattern_detection": {
            "enabled": settings.pattern_detection_enabled,
            "min_pattern_frequency": settings.min_pattern_frequency,
            "pattern_confidence_threshold": settings.pattern_confidence_threshold,
        },
        "performance_monitoring": {
            "metrics_enabled": settings.metrics_enabled,
            "performance_tracking_enabled": settings.performance_tracking_enabled,
            "trend_analysis_enabled": settings.trend_analysis_enabled,
        },
        "database": {
            "url": settings.database_url,
            "results_retention_days": settings.results_retention_days,
        },
        "api": {
            "host": settings.api_host,
            "port": settings.api_port,
            "workers": settings.api_workers,
            "cors_origins": settings.cors_origins,
        },
        "logging": {
            "level": settings.log_level,
            "format": settings.log_format,
            "file": settings.log_file,
        },
        "storage": {
            "data_dir": settings.data_dir,
            "results_dir": settings.results_dir,
            "temp_dir": settings.temp_dir,
            "max_file_size": settings.max_file_size,
        },
        "security": {
            "secret_key": settings.secret_key,
            "enable_cors": settings.enable_cors,
        },
        "development": {
            "debug": settings.debug,
            "testing": settings.testing,
        }
    }


def validate_config() -> bool:
    """
    Validate the current configuration.
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        settings = get_settings()
        
        # Validate confidence threshold
        if not 0.0 <= settings.default_confidence_threshold <= 1.0:
            raise ValueError("default_confidence_threshold must be between 0.0 and 1.0")
        
        # Validate timeout
        if settings.evaluation_timeout <= 0:
            raise ValueError("evaluation_timeout must be positive")
        
        # Validate concurrent evaluations
        if settings.max_concurrent_evaluations <= 0:
            raise ValueError("max_concurrent_evaluations must be positive")
        
        # Validate statistical parameters
        if not 0.0 <= settings.confidence_interval <= 1.0:
            raise ValueError("confidence_interval must be between 0.0 and 1.0")
        
        if not 0.0 <= settings.statistical_significance_threshold <= 1.0:
            raise ValueError("statistical_significance_threshold must be between 0.0 and 1.0")
        
        # Validate pattern detection
        if settings.min_pattern_frequency <= 0:
            raise ValueError("min_pattern_frequency must be positive")
        
        if not 0.0 <= settings.pattern_confidence_threshold <= 1.0:
            raise ValueError("pattern_confidence_threshold must be between 0.0 and 1.0")
        
        # Validate API settings
        if settings.api_port <= 0 or settings.api_port > 65535:
            raise ValueError("api_port must be between 1 and 65535")
        
        if settings.api_workers <= 0:
            raise ValueError("api_workers must be positive")
        
        # Validate file size
        if settings.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
    """
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    # Update settings with new values
    for key, value in updates.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)


def is_development_mode() -> bool:
    """
    Check if running in development mode.
    
    Returns:
        bool: True if in development mode
    """
    settings = get_settings()
    return settings.debug or settings.testing


def get_data_directory() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path: Data directory path
    """
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_results_directory() -> Path:
    """
    Get the results directory path.
    
    Returns:
        Path: Results directory path
    """
    settings = get_settings()
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(exist_ok=True)
    return results_dir


def get_temp_directory() -> Path:
    """
    Get the temporary directory path.
    
    Returns:
        Path: Temporary directory path
    """
    settings = get_settings()
    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def get_log_directory() -> Path:
    """
    Get the log directory path.
    
    Returns:
        Path: Log directory path
    """
    settings = get_settings()
    log_file = Path(settings.log_file)
    log_dir = log_file.parent
    log_dir.mkdir(exist_ok=True)
    return log_dir 