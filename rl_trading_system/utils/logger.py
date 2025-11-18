"""
Module: utils.logger
Description: Unified logging utility for the RL trading system
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

This module provides a centralized logging configuration with support for
file and console output, structured logging, and performance tracking.

Usage Example:
    >>> from rl_trading_system.utils.logger import get_logger, setup_logging
    >>> setup_logging(level='INFO', log_dir='./logs')
    >>> logger = get_logger(__name__)
    >>> logger.info("Trading started")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs

    This formatter is useful for log aggregation systems like ELK or Splunk.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'context'):
            log_data['context'] = record.context

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output

    Colors are based on log level for better readability.
    """

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors

        Args:
            record: Log record to format

        Returns:
            Colored log string
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = 'INFO',
    log_dir: str = './logs',
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
    use_json: bool = False,
    use_colors: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup logging configuration for the entire application

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_format: Custom log format string
        use_json: Whether to use JSON formatting for file logs
        use_colors: Whether to use colored output for console
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep

    Example:
        >>> setup_logging(
        ...     level='DEBUG',
        ...     log_dir='./logs',
        ...     use_json=True
        ... )
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Setup file handler
    if log_to_file:
        log_file = Path(log_dir) / f'trading_{datetime.now().strftime("%Y%m%d")}.log'

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

        if use_json:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(log_format)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Setup console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)

        if use_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format)
        else:
            console_formatter = logging.Formatter(log_format)

        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Setup error file handler (always logs ERROR and CRITICAL)
    error_file = Path(log_dir) / f'error_{datetime.now().strftime("%Y%m%d")}.log'
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(error_handler)

    logging.info(f"Logging initialized - Level: {level}, Dir: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding contextual information to logs

    This allows adding extra fields to all log messages within a context.

    Example:
        >>> with LogContext(trade_id='12345', symbol='BTC/USDT'):
        ...     logger.info("Trade executed")
        # Log will include trade_id and symbol fields
    """

    def __init__(self, **context):
        """
        Initialize log context

        Args:
            **context: Context key-value pairs
        """
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Enter context"""
        old_factory = logging.getLogRecordFactory()
        self.old_factory = old_factory

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.context = self.context
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        logging.setLogRecordFactory(self.old_factory)


class PerformanceLogger:
    """
    Logger for performance metrics

    Tracks execution time and logs performance warnings if thresholds are exceeded.

    Example:
        >>> perf_logger = PerformanceLogger('data_processing')
        >>> with perf_logger:
        ...     # Code to measure
        ...     process_data()
        # Logs: "data_processing completed in 123.45ms"
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[logging.Logger] = None,
        threshold_ms: Optional[float] = None
    ):
        """
        Initialize performance logger

        Args:
            operation: Name of operation being measured
            logger: Logger instance (uses root logger if None)
            threshold_ms: Warning threshold in milliseconds
        """
        self.operation = operation
        self.logger = logger or logging.getLogger()
        self.threshold_ms = threshold_ms
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log"""
        import time
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is None:
            if self.threshold_ms and elapsed_ms > self.threshold_ms:
                self.logger.warning(
                    f"{self.operation} took {elapsed_ms:.2f}ms "
                    f"(threshold: {self.threshold_ms}ms)"
                )
            else:
                self.logger.debug(
                    f"{self.operation} completed in {elapsed_ms:.2f}ms"
                )
        else:
            self.logger.error(
                f"{self.operation} failed after {elapsed_ms:.2f}ms"
            )


class AuditLogger:
    """
    Specialized logger for audit trail

    Records all trading actions to a separate audit log for compliance.

    Example:
        >>> audit_logger = AuditLogger('./logs/audit')
        >>> audit_logger.log_trade(
        ...     action='buy',
        ...     symbol='BTC/USDT',
        ...     quantity=1.0,
        ...     price=50000.0
        ... )
    """

    def __init__(self, log_dir: str = './logs/audit'):
        """
        Initialize audit logger

        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup audit logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Don't propagate to root logger

        # Create handler with daily rotation
        log_file = self.log_dir / f'audit_{datetime.now().strftime("%Y%m%d")}.log'
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def log_trade(
        self,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        **kwargs
    ) -> None:
        """
        Log a trade action

        Args:
            action: Trade action (buy, sell, etc.)
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            **kwargs: Additional trade details
        """
        self.logger.info(
            'Trade executed',
            extra={'context': {
                'action': action,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                **kwargs
            }}
        )

    def log_risk_event(self, event_type: str, severity: str, **kwargs) -> None:
        """
        Log a risk event

        Args:
            event_type: Type of risk event
            severity: Severity level
            **kwargs: Event details
        """
        self.logger.warning(
            f'Risk event: {event_type}',
            extra={'context': {
                'event_type': event_type,
                'severity': severity,
                **kwargs
            }}
        )
