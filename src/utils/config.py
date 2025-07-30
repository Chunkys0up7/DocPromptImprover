"""
Configuration management for the document processing system.

This module handles loading and managing configuration from environment
variables and configuration files.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from pathlib import Path

from ..core.models import OCREngine, OptimizationConfig, QualityControlConfig


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Language Model Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    
    # LLM Provider Selection
    default_llm_provider: str = Field("openai", env="DEFAULT_LLM_PROVIDER")
    default_model: str = Field("gpt-3.5-turbo", env="DEFAULT_MODEL")
    
    # OCR Configuration
    tesseract_path: Optional[str] = Field(None, env="TESSERACT_PATH")
    ocr_engine: OCREngine = Field(OCREngine.TESSERACT, env="OCR_ENGINE")
    
    # Cloud OCR Services
    google_cloud_credentials_path: Optional[str] = Field(None, env="GOOGLE_CLOUD_CREDENTIALS_PATH")
    azure_document_intelligence_endpoint: Optional[str] = Field(None, env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    azure_document_intelligence_key: Optional[str] = Field(None, env="AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./doc_processing.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Processing Configuration
    max_concurrent_processes: int = Field(4, env="MAX_CONCURRENT_PROCESSES")
    default_confidence_threshold: float = Field(0.7, env="DEFAULT_CONFIDENCE_THRESHOLD")
    max_retry_attempts: int = Field(3, env="MAX_RETRY_ATTEMPTS")
    processing_timeout: int = Field(300, env="PROCESSING_TIMEOUT")
    
    # Optimization Configuration
    optimization_enabled: bool = Field(True, env="OPTIMIZATION_ENABLED")
    optimization_interval: int = Field(100, env="OPTIMIZATION_INTERVAL")
    miprov2_num_candidates: int = Field(5, env="MIPROV2_NUM_CANDIDATES")
    miprov2_init_temperature: float = Field(0.5, env="MIPROV2_INIT_TEMPERATURE")
    
    # Quality Control
    assertion_enabled: bool = Field(True, env="ASSERTION_ENABLED")
    quality_check_enabled: bool = Field(True, env="QUALITY_CHECK_ENABLED")
    auto_correction_enabled: bool = Field(True, env="AUTO_CORRECTION_ENABLED")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file: str = Field("logs/doc_processing.log", env="LOG_FILE")
    
    # Performance Monitoring
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    prometheus_port: int = Field(8000, env="PROMETHEUS_PORT")
    
    # Cost Management
    cost_tracking_enabled: bool = Field(True, env="COST_TRACKING_ENABLED")
    max_cost_per_document: float = Field(0.10, env="MAX_COST_PER_DOCUMENT")
    budget_limit: float = Field(100.00, env="BUDGET_LIMIT")
    
    # File Storage
    upload_dir: str = Field("uploads", env="UPLOAD_DIR")
    processed_dir: str = Field("processed", env="PROCESSED_DIR")
    temp_dir: str = Field("temp", env="TEMP_DIR")
    max_file_size: int = Field(10485760, env="MAX_FILE_SIZE")  # 10MB
    
    # Security
    secret_key: str = Field("your_secret_key_here", env="SECRET_KEY")
    allowed_extensions: str = Field("pdf,png,jpg,jpeg,tiff", env="ALLOWED_EXTENSIONS")
    enable_cors: bool = Field(True, env="ENABLE_CORS")
    
    # Development
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    mock_llm_responses: bool = Field(False, env="MOCK_LLM_RESPONSES")
    
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
    
    config = {
        "llm_provider": settings.default_llm_provider,
        "default_model": settings.default_model,
        "ocr_engine": settings.ocr_engine,
        "max_concurrent_processes": settings.max_concurrent_processes,
        "processing_timeout": settings.processing_timeout,
        "optimization": {
            "enabled": settings.optimization_enabled,
            "interval": settings.optimization_interval,
            "num_candidates": settings.miprov2_num_candidates,
            "init_temperature": settings.miprov2_init_temperature,
        },
        "quality_control": {
            "assertion_enabled": settings.assertion_enabled,
            "quality_check_enabled": settings.quality_check_enabled,
            "auto_correction_enabled": settings.auto_correction_enabled,
            "confidence_threshold": settings.default_confidence_threshold,
            "max_retry_attempts": settings.max_retry_attempts,
        },
        "logging": {
            "level": settings.log_level,
            "format": settings.log_format,
            "file": settings.log_file,
        },
        "cost_management": {
            "enabled": settings.cost_tracking_enabled,
            "max_cost_per_document": settings.max_cost_per_document,
            "budget_limit": settings.budget_limit,
        },
        "file_storage": {
            "upload_dir": settings.upload_dir,
            "processed_dir": settings.processed_dir,
            "temp_dir": settings.temp_dir,
            "max_file_size": settings.max_file_size,
            "allowed_extensions": settings.allowed_extensions.split(","),
        },
        "development": {
            "debug": settings.debug,
            "testing": settings.testing,
            "mock_llm_responses": settings.mock_llm_responses,
        }
    }
    
    return config


def validate_config() -> bool:
    """
    Validate the current configuration.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        settings = get_settings()
        
        # Check required API keys based on provider
        if settings.default_llm_provider == "openai" and not settings.openai_api_key:
            print("Warning: OpenAI API key not set")
            return False
        
        if settings.default_llm_provider == "anthropic" and not settings.anthropic_api_key:
            print("Warning: Anthropic API key not set")
            return False
        
        if settings.default_llm_provider == "cohere" and not settings.cohere_api_key:
            print("Warning: Cohere API key not set")
            return False
        
        # Check file paths
        if settings.upload_dir and not os.path.exists(settings.upload_dir):
            os.makedirs(settings.upload_dir, exist_ok=True)
        
        if settings.processed_dir and not os.path.exists(settings.processed_dir):
            os.makedirs(settings.processed_dir, exist_ok=True)
        
        if settings.temp_dir and not os.path.exists(settings.temp_dir):
            os.makedirs(settings.temp_dir, exist_ok=True)
        
        # Check log directory
        log_dir = Path(settings.log_file).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    # Create new settings with updates
    current_settings = get_settings()
    updated_dict = current_settings.dict()
    updated_dict.update(updates)
    
    # Create new settings instance
    _settings = Settings(**updated_dict)


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for specified provider.
    
    Args:
        provider: LLM provider name
        
    Returns:
        Optional[str]: API key if available
    """
    settings = get_settings()
    
    if provider == "openai":
        return settings.openai_api_key
    elif provider == "anthropic":
        return settings.anthropic_api_key
    elif provider == "cohere":
        return settings.cohere_api_key
    else:
        return None


def is_development_mode() -> bool:
    """Check if running in development mode."""
    settings = get_settings()
    return settings.debug or settings.testing


def get_allowed_extensions() -> list:
    """Get list of allowed file extensions."""
    settings = get_settings()
    return [ext.strip() for ext in settings.allowed_extensions.split(",")]


def get_optimization_config() -> OptimizationConfig:
    """Get optimization configuration."""
    settings = get_settings()
    return OptimizationConfig(
        enabled=settings.optimization_enabled,
        interval=settings.optimization_interval,
        num_candidates=settings.miprov2_num_candidates,
        init_temperature=settings.miprov2_init_temperature,
    )


def get_quality_control_config() -> QualityControlConfig:
    """Get quality control configuration."""
    settings = get_settings()
    return QualityControlConfig(
        assertion_enabled=settings.assertion_enabled,
        quality_check_enabled=settings.quality_check_enabled,
        auto_correction_enabled=settings.auto_correction_enabled,
        confidence_threshold=settings.default_confidence_threshold,
        max_retry_attempts=settings.max_retry_attempts,
    ) 