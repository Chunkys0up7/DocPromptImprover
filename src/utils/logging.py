"""
Logging configuration for the document processing system.

This module provides structured logging with different output formats
and configurable log levels for debugging and monitoring.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from datetime import datetime

from .config import get_settings


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format (json, console, or structured)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif format_type == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if enable_console else None,
        level=getattr(logging, level.upper()),
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        structlog.BoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)


class DocumentProcessingLogger:
    """
    Specialized logger for document processing operations.
    
    This logger provides structured logging with specific fields
    for document processing metrics and events.
    """
    
    def __init__(self, name: str = "document_processor"):
        self.logger = get_logger(name)
        self.name = name
    
    def log_document_processing_start(self, 
                                    document_id: str,
                                    document_type: str,
                                    file_path: Optional[str] = None) -> None:
        """Log the start of document processing."""
        self.logger.info(
            "Document processing started",
            document_id=document_id,
            document_type=document_type,
            file_path=file_path,
            event="processing_start"
        )
    
    def log_document_processing_complete(self,
                                       document_id: str,
                                       success_rate: float,
                                       processing_time: float,
                                       confidence: float) -> None:
        """Log the completion of document processing."""
        self.logger.info(
            "Document processing completed",
            document_id=document_id,
            success_rate=success_rate,
            processing_time=processing_time,
            confidence=confidence,
            event="processing_complete"
        )
    
    def log_field_extraction(self,
                           document_id: str,
                           field_name: str,
                           status: str,
                           confidence: float,
                           raw_value: Optional[str] = None) -> None:
        """Log individual field extraction results."""
        self.logger.debug(
            "Field extraction result",
            document_id=document_id,
            field_name=field_name,
            status=status,
            confidence=confidence,
            raw_value=raw_value,
            event="field_extraction"
        )
    
    def log_optimization_trigger(self,
                               trigger_type: str,
                               documents_processed: int,
                               current_metrics: Dict[str, float]) -> None:
        """Log optimization trigger events."""
        self.logger.info(
            "Optimization triggered",
            trigger_type=trigger_type,
            documents_processed=documents_processed,
            current_metrics=current_metrics,
            event="optimization_trigger"
        )
    
    def log_optimization_complete(self,
                                optimization_id: str,
                                improvement_metrics: Dict[str, float],
                                duration: float) -> None:
        """Log optimization completion."""
        self.logger.info(
            "Optimization completed",
            optimization_id=optimization_id,
            improvement_metrics=improvement_metrics,
            duration=duration,
            event="optimization_complete"
        )
    
    def log_error(self,
                 error_type: str,
                 error_message: str,
                 document_id: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Log error events."""
        self.logger.error(
            "Processing error occurred",
            error_type=error_type,
            error_message=error_message,
            document_id=document_id,
            context=context or {},
            event="processing_error"
        )
    
    def log_performance_metric(self,
                             metric_name: str,
                             metric_value: float,
                             document_id: Optional[str] = None) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            metric_value=metric_value,
            document_id=document_id,
            event="performance_metric"
        )
    
    def log_quality_check(self,
                         document_id: str,
                         quality_score: float,
                         issues: list,
                         suggestions: list) -> None:
        """Log quality check results."""
        self.logger.info(
            "Quality check completed",
            document_id=document_id,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            event="quality_check"
        )


class MetricsLogger:
    """
    Logger specifically for metrics and statistics.
    
    This logger is optimized for high-frequency metric logging
    and can be configured for different output formats.
    """
    
    def __init__(self, name: str = "metrics"):
        self.logger = get_logger(name)
        self.name = name
    
    def log_field_success_rate(self,
                              field_name: str,
                              success_rate: float,
                              sample_size: int) -> None:
        """Log field-level success rates."""
        self.logger.info(
            "Field success rate",
            field_name=field_name,
            success_rate=success_rate,
            sample_size=sample_size,
            event="field_success_rate"
        )
    
    def log_overall_metrics(self,
                           total_documents: int,
                           success_rate: float,
                           average_processing_time: float,
                           error_rate: float) -> None:
        """Log overall processing metrics."""
        self.logger.info(
            "Overall processing metrics",
            total_documents=total_documents,
            success_rate=success_rate,
            average_processing_time=average_processing_time,
            error_rate=error_rate,
            event="overall_metrics"
        )
    
    def log_cost_metrics(self,
                        total_cost: float,
                        cost_per_document: float,
                        documents_processed: int) -> None:
        """Log cost-related metrics."""
        self.logger.info(
            "Cost metrics",
            total_cost=total_cost,
            cost_per_document=cost_per_document,
            documents_processed=documents_processed,
            event="cost_metrics"
        )
    
    def log_confidence_distribution(self,
                                  high_confidence: int,
                                  medium_confidence: int,
                                  low_confidence: int) -> None:
        """Log confidence score distribution."""
        self.logger.info(
            "Confidence distribution",
            high_confidence=high_confidence,
            medium_confidence=medium_confidence,
            low_confidence=low_confidence,
            event="confidence_distribution"
        )


def initialize_logging() -> None:
    """Initialize logging based on configuration."""
    settings = get_settings()
    
    setup_logging(
        level=settings.log_level,
        format_type=settings.log_format,
        log_file=settings.log_file,
        enable_console=True
    )


def create_processing_logger(name: str = "document_processor") -> DocumentProcessingLogger:
    """Create a document processing logger instance."""
    return DocumentProcessingLogger(name)


def create_metrics_logger(name: str = "metrics") -> MetricsLogger:
    """Create a metrics logger instance."""
    return MetricsLogger(name)


# Global logger instances
_processing_logger: Optional[DocumentProcessingLogger] = None
_metrics_logger: Optional[MetricsLogger] = None


def get_processing_logger() -> DocumentProcessingLogger:
    """Get the global processing logger instance."""
    global _processing_logger
    if _processing_logger is None:
        _processing_logger = DocumentProcessingLogger()
    return _processing_logger


def get_metrics_logger() -> MetricsLogger:
    """Get the global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger


# Initialize logging when module is imported
try:
    initialize_logging()
except Exception as e:
    # Fallback to basic logging if configuration fails
    logging.basicConfig(level=logging.INFO)
    logging.warning(f"Failed to initialize structured logging: {e}") 