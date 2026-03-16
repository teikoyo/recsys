"""Logging setup for WS-SGNS."""

import logging
import sys


def get_logger(name: str = "ws_sgns") -> logging.Logger:
    """Get or create a logger with console handler.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


def log_rank0(
    logger: logging.Logger,
    is_ddp: bool,
    rank: int,
    msg: str,
    level: int = logging.INFO,
) -> None:
    """Log a message only from rank 0 in DDP mode.

    Args:
        logger: Logger instance.
        is_ddp: Whether running in DDP mode.
        rank: Current process rank.
        msg: Message to log.
        level: Logging level.
    """
    if (not is_ddp) or rank == 0:
        logger.log(level, msg)
